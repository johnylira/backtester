import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt

def log(msg):
    print(f"[LOG] {msg}")

# Baixar os dados
url = "https://n8n.coinverge.com.br/webhook/precos"
log("Baixando dados...")
response = requests.get(url)
data = response.json()
log(f"Dados baixados: {len(data)} registros")

# DataFrame
df = pd.DataFrame(data)
log(f"DataFrame criado com colunas: {df.columns.tolist()} e shape: {df.shape}")

# Conversão numérica
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except:
        pass

# Garantir datetime para ordenar corretamente (remover timezone)
if 'datahora_fechamento' in df.columns:
    try:
        df['datahora_fechamento'] = pd.to_datetime(df['datahora_fechamento']).dt.tz_localize(None)
    except Exception as e:
        log(f"Erro convertendo 'datahora_fechamento': {e}")

class BacktestStopDinamicoTodos:
    def __init__(self, df):
        self.df = df.copy()
        self.resultados = {}

    def simular(self):
        ultimas_variacoes = []
        simbolos = self.df['simbolo'].unique()
        for simbolo in simbolos:
            grupo = self.df[self.df['simbolo'] == simbolo].copy()
            if len(grupo) < 24:
                continue
            grupo = grupo.sort_values('datahora_fechamento')

            grupo['var'] = ((grupo['ultimo_preco'] / grupo['ultimo_preco'].shift(1) - 1) - 0.000575 - 0.001) * 0.85
            grupo['max_desceu'] = (grupo['preco_minimo'] / grupo['ultimo_preco'].shift(1) - 1)
            grupo['stop'] = np.where(
                (grupo['var'].shift(1).fillna(0) < 0),
                0,
                grupo['max_desceu'].shift(1).fillna(0)
            )
            grupo['retorno_estrategia'] = np.where(
                grupo['max_desceu'] < grupo['stop'],
                (grupo['stop'] - 0.000576 - 0.001) * 0.85,
                grupo['var']
            )
            grupo['acumulado_estrategia'] = grupo['retorno_estrategia'].cumsum()
            grupo['acumulado_var'] = grupo['var'].cumsum()
            self.resultados[simbolo] = grupo

            ultima_var = grupo.iloc[-1]['retorno_estrategia']
            print(f"Processando {simbolo}: última variação = {ultima_var:.6f}, stop = {grupo.iloc[-1]['stop']:.6f}")
            ultimas_variacoes.append({'simbolo': simbolo, 'ultima_var': ultima_var, 'stop': grupo.iloc[-1]['stop']})

        if not ultimas_variacoes:
            log("Nenhum ativo processado")
            return None

        df_variacoes = pd.DataFrame(ultimas_variacoes)
        melhor_idx = df_variacoes['stop'].idxmax()
        melhor_ativo = df_variacoes.loc[melhor_idx]
        log(f"Melhor ativo do período: {melhor_ativo['simbolo']} com variação {melhor_ativo['ultima_var']:.6f}")
        return melhor_ativo

    def estrategia_multiativo(self):
        # Construir DataFrame concatenando todos os ativos
        todos = []
        for simbolo, grupo in self.resultados.items():
            df_aux = grupo[['datahora_fechamento', 'retorno_estrategia']].copy()
            df_aux['simbolo'] = simbolo
            todos.append(df_aux)

        df_todos = pd.concat(todos)

        # Garantir unicidade para pivot
        df_todos_agg = df_todos.groupby(['datahora_fechamento', 'simbolo'], as_index=False).agg({'retorno_estrategia': 'last'})

        # Pivot para matriz de retornos por data e ativo
        tabela = df_todos_agg.pivot(index='datahora_fechamento', columns='simbolo', values='retorno_estrategia').sort_index()

        # Ativo com maior retorno por hora e seus retornos
        escolhas = tabela.idxmax(axis=1)  # símbolo escolhido
        retornos_multiativo = tabela.max(axis=1).fillna(0)
        acumulado_multiativo = retornos_multiativo.cumsum()

        # Criar tabela informativa detalhada
        info_df = tabela.copy()
        info_df['ativo_recomendado'] = escolhas
        info_df['retorno_recomendado'] = retornos_multiativo

        # Verificar se recomendação bate com maior retorno por hora (sempre deve bater)
        # Vamos comparar ativamente ativo recomendado vs máximo da linha - trivialmente True
        info_df['confere_recomendacao'] = True  # Como idxmax garante isso

        # Mudar organização para mostrar colunas mais importantes claramente
        # Também incluir coluna com variação normal (acumulado var) do ativo recomendado, usando self.resultados
        acumulados_normais = []
        for tempo, ativo in zip(info_df.index, info_df['ativo_recomendado']):
            if ativo in self.resultados:
                df_ativo = self.resultados[ativo]
                # Achar o índice da datahora_fechamento igual a tempo para pegar valor acumulado_var
                linha = df_ativo[df_ativo['datahora_fechamento'] == tempo]
                if not linha.empty:
                    acumulados_normais.append(linha['acumulado_var'].values[0])
                else:
                    acumulados_normais.append(np.nan)
            else:
                acumulados_normais.append(np.nan)
        info_df['acumulado_var_ativo_recomendado'] = acumulados_normais

        # Resetar índice para exportar e imprimir
        info_df_reset = info_df.reset_index()

        # Identificando o primeiro ativo recomendado (primeira linha da série temporal)
        primeiro_ativo = info_df_reset.loc[0, 'ativo_recomendado']
        log(f"Primeiro ativo recomendado: {primeiro_ativo}")

        # Exportar tabela informativa para Excel
        arquivo_excel = 'relatorio_detalhado_recomendacoes.xlsx'
        info_df_reset.to_excel(arquivo_excel, index=False)
        log(f"Exportação para Excel concluída: {arquivo_excel}")

        return info_df_reset, primeiro_ativo

# Executar backtest
backtest = BacktestStopDinamicoTodos(df)
melhor_do_periodo = backtest.simular()

print("\nResultado final:")
if melhor_do_periodo is not None:
    simbolo_escolhido = melhor_do_periodo['simbolo']
    variacao_escolhida = melhor_do_periodo['ultima_var']
    print(f"Ativo selecionado: {simbolo_escolhido} - Variação: {variacao_escolhida:.6f}")

    tabela_info, primeiro_ativo = backtest.estrategia_multiativo()

    print(f"\nPrimeiro ativo recomendado: {primeiro_ativo}")
    print("\nTabela detalhada das recomendações e variações (primeiras 10 linhas):")
    print(tabela_info.head(10).to_string())

    # Plot comparativo (opcional)
    df_ativo = backtest.resultados[simbolo_escolhido]
    plt.figure(figsize=(14,7))
    plt.plot(df_ativo['datahora_fechamento'], df_ativo['acumulado_var'], label='Acumulado Variação Normal', linewidth=2)
    plt.plot(df_ativo['datahora_fechamento'], df_ativo['acumulado_estrategia'], label='Acumulado Estratégia Stop', linewidth=2)
    plt.plot(tabela_info['datahora_fechamento'], tabela_info['retorno_recomendado'].cumsum(),
             label='Multiativo (acumulado retorno recomendado)', linewidth=2, linestyle='--')
    plt.title(f'Comparativo Estratégias - Ativo: {simbolo_escolhido} vs Estratégia Multiativo')
    plt.xlabel('Data/Hora de Fechamento')
    plt.ylabel('Variação Acumulada')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Nenhum ativo disponível para seleção!")
