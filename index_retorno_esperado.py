import os
import sys
import math
import json
import time
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, Dict, List

# =======================
# Parâmetros editáveis
# =======================
PARAMS = {
    # Entrada de dados
    "api_url": "https://n8n.coinverge.com.br/webhook/precos",

    # Seleção e sinais
    "k": 10,                     # top-k
    "suporte_min": 30,           # ocorrências mínimas do cenário
    "cap_ativo": 0.10,           # teto de peso por ativo (10%)
    "kelly_frac": 0.5,           # half-Kelly
    "slippage_bps": 10,          # 10 bps por trade => 20 bps roundtrip

    # Risco
    "vol_alvo": 0.15,            # vol anualizada alvo do portfólio
    "rolling_cov_days": 60,      # janela para estimar covariância ex-ante
    "dd_cut": 0.10,              # drawdown gatilho (10%)
    "initial_equity": 1_000_000, # patrimônio inicial p/ curva de DD

    # Pastas/arquivos
    "dir_data": "data",
    "dir_excel": "excel",
    "file_retornos_excel": "excel/retorno_esperado_todos.xlsx",
    "file_retornos_csv": "data/retorno_esperado_todos.csv",
    "file_carteira_csv": "data/carteira_alvo.csv",
    "file_carteira_xlsx": "excel/carteira_alvo.xlsx",
    "file_equity_curve": "data/equity_curve.csv",
}

# =======================
# Utilidades de I/O e validação
# =======================
def ensure_dirs(params: Dict):
    Path(params["dir_data"]).mkdir(parents=True, exist_ok=True)
    Path(params["dir_excel"]).mkdir(parents=True, exist_ok=True)

def safe_request_json(url: str, timeout: int = 20, retries: int = 3, backoff: int = 3):
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff * (attempt + 1))
    raise RuntimeError(f"Falha ao obter JSON da API {url}: {last_err}")

def ensure_datetime_no_tz(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        # Se vier timezone-aware, remover
        if getattr(df[col].dt, 'tz', None) is not None:
            df[col] = df[col].dt.tz_localize(None)
    return df

def coerce_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    for c in columns:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

# =======================
# 1) Ingestão da API e preparação de preços
# =======================
def load_prices_from_api(params: Dict) -> pd.DataFrame:
    data = safe_request_json(params["api_url"])
    df = pd.DataFrame(data)

    # Campos mínimos esperados
    req_cols = {"simbolo", "datahora_fechamento", "ultimo_preco"}
    missing = req_cols - set(df.columns)
    if missing:
        raise ValueError(f"API retornou dados sem colunas necessárias: {missing}")

    # Conversões
    df = ensure_datetime_no_tz(df, "datahora_fechamento")
    df = coerce_numeric(df, ["ultimo_preco"])

    # Limpeza básica
    df = df.dropna(subset=["simbolo", "datahora_fechamento", "ultimo_preco"])
    df["simbolo"] = df["simbolo"].astype(str).str.strip()
    df = df.sort_values(["simbolo", "datahora_fechamento"]).reset_index(drop=True)

    # Remover duplicatas exatas
    df = df.drop_duplicates(subset=["simbolo", "datahora_fechamento"], keep="last")

    return df

def compute_daily_returns(df_prices: pd.DataFrame) -> pd.DataFrame:
    dfp = df_prices.copy()
    dfp = dfp.sort_values(["simbolo", "datahora_fechamento"])
    dfp["ret"] = dfp.groupby("simbolo")["ultimo_preco"].pct_change()
    return dfp

# =======================
# 2) Pipeline RetornoEsperado (reimplementado)
# =======================
class RetornoEsperado:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.resultados = None

    def calcular_retorno_esperado(self, max_window: int = None):
        """
        Constrói cenários de altas consecutivas:
        - Para cada símbolo, para todas as janelas de tamanho n (2..N ou 2..max_window),
          se todos os retornos da janela forem > 0, registra o retorno do próximo dia.
        - Agrega colunas: simbolo, cenario ('+'*n), retorno_proximo, ganhos, perdas, binario, dia1..diaN.
        """
        if self.df.empty:
            print("DataFrame de preços vazio.")
            self.resultados = pd.DataFrame()
            return

        req_cols = {"simbolo", "datahora_fechamento", "ultimo_preco"}
        missing = req_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"df faltando colunas necessárias: {missing}")

        # Garantir ordenação e retornos diários
        dfp = self.df.copy()
        dfp = dfp.sort_values(["simbolo", "datahora_fechamento"])
        dfp["ret"] = dfp.groupby("simbolo")["ultimo_preco"].pct_change()

        simbolos = dfp["simbolo"].unique()
        todos_cenarios = []

        for simbolo in simbolos:
            grupo = dfp[dfp["simbolo"] == simbolo].copy().reset_index(drop=True)
            grupo = grupo.dropna(subset=["ret"]).reset_index(drop=True)
            if grupo.empty:
                continue

            N = len(grupo)
            upper = N if max_window is None else min(max_window, N)

            # Para cada janela
            for n in range(2, upper):
                # índice i finaliza em i+n, então até N-n-1
                for i in range(0, len(grupo) - n):
                    janela = grupo.iloc[i:i+n]
                    if not (janela["ret"] > 0).all():
                        continue
                    # Próximo dia
                    if i + n >= len(grupo):
                        continue
                    proximo = grupo.iloc[i + n]
                    retorno = proximo["ret"]
                    ganhos = retorno if pd.notna(retorno) and retorno > 0 else np.nan
                    perdas = retorno if pd.notna(retorno) and retorno < 0 else np.nan
                    binario = 1 if pd.notna(retorno) and retorno > 0 else 0

                    linha = {
                        "simbolo": simbolo,
                        "cenario": "+" * n,
                        "retorno_proximo": retorno,
                        "ganhos": ganhos,
                        "perdas": perdas,
                        "binario": binario
                    }
                    # Adicionar variáveis da janela
                    for j in range(n):
                        linha[f"dia{j+1}"] = janela.iloc[j]["ret"]

                    todos_cenarios.append(linha)

        if not todos_cenarios:
            print("Nenhum cenário encontrado no histórico.")
            self.resultados = pd.DataFrame(columns=["simbolo","cenario","retorno_proximo","ganhos","perdas","binario"])
            return

        df_cenario = pd.DataFrame(todos_cenarios)

        # Calcular média dos binários por ativo (media_binario), se for útil
        medias_binarios = df_cenario.groupby("simbolo")["binario"].mean().reset_index()
        medias_binarios.rename(columns={"binario": "media_binario"}, inplace=True)

        df_final = pd.merge(df_cenario, medias_binarios, on="simbolo", how="left")
        self.resultados = df_final

    def salvar_excel_unico(self, path_excel: str):
        if self.resultados is None or self.resultados.empty:
            print("Nenhum resultado para salvar (RetornoEsperado).")
            return
        Path(os.path.dirname(path_excel)).mkdir(parents=True, exist_ok=True)
        self.resultados.to_excel(path_excel, index=False)
        print(f"Arquivo único salvo: {path_excel}")

# =======================
# 3) Agregação de sinais por (simbolo, cenario)
# =======================
def prepare_signal_table(resultados: pd.DataFrame, params: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    logs = []

    req_cols = {"simbolo", "cenario", "retorno_proximo", "ganhos", "perdas", "binario"}
    missing = req_cols - set(resultados.columns)
    if missing:
        raise ValueError(f"resultados faltando colunas: {missing}")

    r = resultados.copy()
    for c in ["retorno_proximo", "ganhos", "perdas", "binario"]:
        r[c] = pd.to_numeric(r[c], errors="coerce")

    agg = r.groupby(["simbolo", "cenario"]).agg(
        p=("binario", "mean"),
        ganho_medio=("ganhos", "mean"),
        perda_media=("perdas", "mean"),
        suporte=("binario", "count"),
        Er=("retorno_proximo", "mean")
    ).reset_index()

    # b válido apenas se perda_media for negativa
    agg["b"] = np.where(
        (~agg["ganho_medio"].isna()) & (~agg["perda_media"].isna()) & (agg["perda_media"] < 0),
        agg["ganho_medio"] / agg["perda_media"].abs(),
        np.nan
    )

    filtro = (
        (agg["suporte"] >= params["suporte_min"]) &
        (agg["p"] > 0.5) &
        (agg["ganho_medio"] > 0) &
        (~agg["b"].isna()) &
        (agg["b"] > 0)
    )
    agg_f = agg[filtro].copy()

    # Logs de descartes por agregação
    discarded = agg[~filtro]
    for _, row in discarded.iterrows():
        motivo = []
        if row["suporte"] < params["suporte_min"]:
            motivo.append(f"suporte<{params['suporte_min']}")
        if row["p"] <= 0.5:
            motivo.append("p<=0.5")
        if pd.isna(row["ganho_medio"]) or row["ganho_medio"] <= 0:
            motivo.append("ganho_medio<=0/NA")
        if pd.isna(row["b"]) or row["b"] <= 0:
            motivo.append("b<=0/NA")
        logs.append(f"{row['simbolo']}|{row['cenario']}: descartado na agregação ({', '.join(motivo)})")

    return agg_f, agg, logs

# =======================
# 4) Detecção do cenário atual por símbolo
# =======================
def detect_current_scenario(dfp: pd.DataFrame) -> pd.DataFrame:
    scenarios = []
    for sym, g in dfp.groupby("simbolo"):
        g = g.sort_values("datahora_fechamento")
        if g.shape[0] < 3:
            scenarios.append({"simbolo": sym, "n": 0, "cenario_atual": None, "data_ref": g["datahora_fechamento"].max()})
            continue
        t = g["datahora_fechamento"].max()
        g_minus1 = g[g["datahora_fechamento"] < t]
        if g_minus1.empty:
            scenarios.append({"simbolo": sym, "n": 0, "cenario_atual": None, "data_ref": t})
            continue

        ret_series = g_minus1["ret"].dropna().values
        n = 0
        for r in ret_series[::-1]:
            if r > 0:
                n += 1
            else:
                break
        cenario = ("+" * n) if n > 0 else None
        scenarios.append({"simbolo": sym, "n": n, "cenario_atual": cenario, "data_ref": t})

    return pd.DataFrame(scenarios)

# =======================
# 5) Score, seleção, sizing e risco
# =======================
def cross_section_scores(scen_df: pd.DataFrame, agg_f: pd.DataFrame, slippage_bps: int) -> Tuple[pd.DataFrame, List[str]]:
    logs = []
    joined = scen_df.merge(
        agg_f,
        left_on=["simbolo", "cenario_atual"],
        right_on=["simbolo", "cenario"],
        how="left",
        suffixes=("", "_agg")
    )

    roundtrip_cost = 2 * (slippage_bps / 10000)  # entra+saí
    joined["Er_net"] = joined["Er"] - roundtrip_cost
    joined["score"] = joined["p"] * joined["Er_net"]

    # Logs de descartes na etapa final
    for _, row in joined.iterrows():
        sym = row["simbolo"]
        if pd.isna(row["cenario_atual"]):
            logs.append(f"{sym}: descartado (sem cenário atual).")
        elif pd.isna(row["p"]) or pd.isna(row["Er"]) or pd.isna(row["b"]):
            logs.append(f"{sym}: descartado (par sem parâmetros suficientes).")
        elif row["Er_net"] <= 0:
            logs.append(f"{sym}: descartado (Er_net<=0).")

    sel = joined[
        (~joined["cenario_atual"].isna()) &
        (~joined["p"].isna()) &
        (~joined["b"].isna()) &
        (~joined["Er_net"].isna()) &
        (joined["Er_net"] > 0)
    ].copy()

    return sel, logs

def select_top_k(sel: pd.DataFrame, k: int) -> pd.DataFrame:
    sel = sel.sort_values("score", ascending=False).copy()
    return sel.head(k).copy()

def kelly_sizing(df_sel: pd.DataFrame, params: Dict) -> pd.DataFrame:
    p = df_sel["p"].values
    b = df_sel["b"].values
    f_star = (b * p - (1 - p)) / b
    f_star = np.maximum(0.0, f_star)
    f_adj = params["kelly_frac"] * f_star
    f_adj = np.minimum(f_adj, params["cap_ativo"])
    out = df_sel.copy()
    out["peso_raw"] = f_adj
    return out

def normalize_weights_long_only(df_sel: pd.DataFrame, target_sum: float = 1.0) -> pd.DataFrame:
    total = df_sel["peso_raw"].sum()
    out = df_sel.copy()
    if total <= 0:
        out["peso"] = 0.0
    else:
        out["peso"] = out["peso_raw"] * (target_sum / total)
    return out

def estimate_cov_matrix(dfp: pd.DataFrame, symbols: List[str], end_date: pd.Timestamp, lookback_days: int) -> pd.DataFrame:
    df_sub = dfp[(dfp["datahora_fechamento"] <= end_date)].copy()
    pivot = df_sub.pivot_table(index="datahora_fechamento", columns="simbolo", values="ret", aggfunc="last")
    cols = [c for c in symbols if c in pivot.columns]
    if not cols:
        return pd.DataFrame()
    pivot = pivot[cols]
    pivot = pivot.dropna(how="all").tail(lookback_days)
    if pivot.shape[0] < 20:
        return pd.DataFrame()
    cov = pivot.cov(min_periods=20)
    return cov

def apply_vol_target(weights: pd.Series, cov_daily: pd.DataFrame, vol_target_annual: float) -> Tuple[pd.Series, float]:
    if cov_daily is None or cov_daily.empty or weights.empty:
        return weights, float("nan")
    w = weights.reindex(cov_daily.index).fillna(0.0).values
    try:
        vol_daily = float(np.sqrt(np.dot(w, np.dot(cov_daily.values, w))))
    except Exception:
        return weights, float("nan")
    vol_annual = vol_daily * np.sqrt(252)
    if not np.isfinite(vol_annual) or vol_annual <= 0:
        return weights, vol_annual
    scale = min(1.0, vol_target_annual / vol_annual)  # só reduz risco
    weights_scaled = weights * scale
    return weights_scaled, vol_annual

def load_or_init_equity_curve(file_path: str, initial_equity: float) -> pd.DataFrame:
    Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    if os.path.exists(file_path):
        eq = pd.read_csv(file_path)
        eq["date"] = pd.to_datetime(eq["date"])
        eq["equity"] = pd.to_numeric(eq["equity"], errors="coerce")
        eq = eq.dropna(subset=["date", "equity"]).sort_values("date")
        return eq
    else:
        eq = pd.DataFrame({"date": [pd.Timestamp("2000-01-01")], "equity": [initial_equity]})
        eq.to_csv(file_path, index=False)
        return eq

def current_drawdown(eq: pd.DataFrame) -> float:
    eq = eq.sort_values("date").copy()
    eq["peak"] = eq["equity"].cummax()
    eq["dd"] = (eq["equity"] - eq["peak"]) / eq["peak"]
    return float(eq["dd"].iloc[-1])

def apply_dd_control(weights: pd.Series, eq: pd.DataFrame, dd_cut: float) -> Tuple[pd.Series, float]:
    dd = current_drawdown(eq)
    if dd <= dd_cut:
        return weights, dd
    return weights * 0.5, dd

# =======================
# 6) Construção da carteira-alvo
# =======================
def build_portfolio_target(
    df_prices: pd.DataFrame,
    resultados: pd.DataFrame,
    params: Dict
) -> Tuple[pd.DataFrame, Dict, List[str]]:
    logs = []

    # Retornos diários
    dfp = compute_daily_returns(df_prices)

    # Agregação de sinais
    agg_f, agg_raw, logs_agg = prepare_signal_table(resultados, params)
    logs.extend(logs_agg)

    # Cenário atual
    scen_df = detect_current_scenario(dfp)
    data_ref = scen_df["data_ref"].max()

    # Scores e custos
    sel_full, logs_sel = cross_section_scores(scen_df, agg_f, params["slippage_bps"])
    logs.extend(logs_sel)

    # Seleção top-k
    topk = select_top_k(sel_full, params["k"])

    # Kelly
    if topk.empty:
        logs.append("Nenhum ativo passou nos filtros finais para seleção.")
        carteira = pd.DataFrame(columns=["data", "simbolo", "peso", "p", "b", "score", "cenario", "suporte"])
        meta = {
            "data": data_ref,
            "num_sinais": int(sel_full.shape[0]),
            "num_selecionados": 0,
            "vol_antes": None,
            "vol_depois": None,
            "dd_atual": None,
            "observacoes": "; ".join(sorted(set(logs)))
        }
        return carteira, meta, logs

    topk_kelly = kelly_sizing(topk, params)
    topk_norm = normalize_weights_long_only(topk_kelly, target_sum=1.0)

    # Risco: Drawdown
    eq = load_or_init_equity_curve(params["file_equity_curve"], params["initial_equity"])
    weights_series = pd.Series(topk_norm.set_index("simbolo")["peso"])
    weights_dd, dd_atual = apply_dd_control(weights_series, eq, params["dd_cut"])

    # Risco: Volatilidade
    symbols = list(weights_dd.index)
    cov_daily = estimate_cov_matrix(dfp, symbols, data_ref, params["rolling_cov_days"])
    weights_vol_scaled, vol_before = apply_vol_target(weights_dd, cov_daily, params["vol_alvo"])

    # Calcular vol após o scale (para registro)
    vol_after = None
    if isinstance(vol_before, float) and np.isfinite(vol_before) and not math.isnan(vol_before) and not (cov_daily is None or cov_daily.empty):
        w_after = weights_vol_scaled.reindex(cov_daily.index).fillna(0.0).values
        try:
            vol_after_daily = float(np.sqrt(np.dot(w_after, np.dot(cov_daily.values, w_after))))
            vol_after = vol_after_daily * np.sqrt(252)
        except Exception:
            vol_after = None

    # Não renormalizamos para 1.0 após downsizing para respeitar redução de risco.
    final_weights = weights_vol_scaled.clip(lower=0.0, upper=params["cap_ativo"])

    # Montar DataFrame final
    out = topk_norm.set_index("simbolo").copy()
    out["peso"] = final_weights.reindex(out.index).fillna(0.0)
    out["data"] = pd.to_datetime(data_ref).normalize()
    out = out.reset_index()

    carteira_alvo = out[["data", "simbolo", "peso", "p", "b", "score", "cenario", "suporte"]].copy()
    carteira_alvo["peso"] = carteira_alvo["peso"].astype(float)

    num_sinais = int(sel_full.shape[0])
    num_sel = int(carteira_alvo.shape)
    logs.append(f"Ativos com sinal: {num_sinais}. Selecionados (top-k): {num_sel}.")
    logs.append(f"Parâmetros: k={params['k']}, suporte_min={params['suporte_min']}, cap_ativo={params['cap_ativo']}, vol_alvo={params['vol_alvo']}, dd_cut={params['dd_cut']}, kelly_frac={params['kelly_frac']}, slippage_bps={params['slippage_bps']}.")

    meta = {
        "data": data_ref,
        "num_sinais": num_sinais,
        "num_selecionados": num_sel,
        "vol_antes": vol_before,
        "vol_depois": vol_after,
        "dd_atual": dd_atual,
        "observacoes": "; ".join(sorted(set(logs)))
    }

    return carteira_alvo, meta, logs

# =======================
# 7) Persistência de saídas e logs
# =======================
def save_retornos_outputs(resultados: pd.DataFrame, params: Dict):
    if resultados is None or resultados.empty:
        print("Aviso: resultados do RetornoEsperado vazios; nada salvo.")
        return
    Path(params["dir_excel"]).mkdir(parents=True, exist_ok=True)
    Path(params["dir_data"]).mkdir(parents=True, exist_ok=True)
    resultados.to_excel(params["file_retornos_excel"], index=False)
    resultados.to_csv(params["file_retornos_csv"], index=False)
    print(f"Salvo RetornoEsperado: {params['file_retornos_excel']} e {params['file_retornos_csv']}")

def save_portfolio_outputs(carteira: pd.DataFrame, meta: Dict, params: Dict):
    Path(params["dir_excel"]).mkdir(parents=True, exist_ok=True)
    Path(params["dir_data"]).mkdir(parents=True, exist_ok=True)
    carteira.to_csv(params["file_carteira_csv"], index=False)
    with pd.ExcelWriter(params["file_carteira_xlsx"]) as writer:
        carteira.to_excel(writer, index=False, sheet_name="carteira")
        pd.DataFrame([meta]).to_excel(writer, index=False, sheet_name="meta")
    print(f"Salvo Carteira: {params['file_carteira_csv']} e {params['file_carteira_xlsx']}")

def print_summary(carteira: pd.DataFrame, meta: Dict, logs: List[str], top_show: int = 10):
    print("\n===== Resumo da Carteira-Alvo =====")
    print(f"Data de referência: {meta.get('data')}")
    print(f"Ativos com sinal: {meta.get('num_sinais')}, Selecionados: {meta.get('num_selecionados')}")
    print(f"Drawdown atual: {meta.get('dd_atual')}")
    print(f"Vol ex-ante antes: {meta.get('vol_antes')}, depois: {meta.get('vol_depois')}")
    print("\nTop ativos e métricas:")
    if not carteira.empty:
        cols = ["simbolo", "peso", "p", "b", "score", "cenario", "suporte"]
        show = carteira.sort_values("peso", ascending=False)[cols].head(top_show).copy()
        for c in ["peso", "p", "b", "score"]:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce").round(6)
        print(show.to_string(index=False))
    else:
        print("Nenhum ativo selecionado.")

    # Logs resumidos
    print("\n===== Logs =====")
    # Limitar para evitar excesso
    uniq = sorted(set(logs))
    for line in uniq[:200]:
        print(line)
    if len(uniq) > 200:
        print(f"... ({len(uniq)-200} linhas adicionais)")

# =======================
# 8) Execução principal end-to-end
# =======================
def main(params: Dict = PARAMS):
    ensure_dirs(params)

    # 1) Baixar API e preparar df de preços
    df = load_prices_from_api(params)

    # 2) Executar RetornoEsperado
    estrategia = RetornoEsperado(df)
    estrategia.calcular_retorno_esperado()
    resultados = estrategia.resultados if estrategia.resultados is not None else pd.DataFrame()

    # 3) Salvar saídas do RetornoEsperado
    save_retornos_outputs(resultados, params)

    # 4) Construir carteira-alvo
    carteira_alvo, meta, logs = build_portfolio_target(df, resultados, params)

    # 5) Persistir carteira
    save_portfolio_outputs(carteira_alvo, meta, params)

    # 6) Imprimir resumo
    print_summary(carteira_alvo, meta, logs, top_show=params["k"])

    # 7) Observação sobre curva patrimonial
    # Para o controle de DD funcionar plenamente ao longo do tempo, alimente "data/equity_curve.csv"
    # diariamente com o patrimônio realizado após executar a carteira. Exemplo de update externo:
    # eq = load_or_init_equity_curve(params["file_equity_curve"], params["initial_equity"])
    # eq = pd.concat([eq, pd.DataFrame({"date":[pd.Timestamp(meta["data"])], "equity":[novo_patrimonio]})])
    # eq = eq.drop_duplicates(subset=["date"]).sort_values("date")
    # eq.to_csv(params["file_equity_curve"], index=False)

if __name__ == "__main__":
    try:
        main(PARAMS)
    except Exception as e:
        print(f"Falha na execução: {e}")
        raise
