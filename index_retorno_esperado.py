import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt

# Parâmetros
LIMIT_SUBIDA_MAXIMA = 0.05

# Baixar os dados pela URL
url = "https://n8n.coinverge.com.br/webhook/precos"
response = requests.get(url)
data = response.json()

# Criar DataFrame
df = pd.DataFrame(data)

# Conversão numérica
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except Exception:
        pass

# Garantir datetime para ordenar corretamente e sem timezone
if 'datahora_fechamento' in df.columns:
    df['datahora_fechamento'] = pd.to_datetime(df['datahora_fechamento']).dt.tz_localize(None)

class EstrategiaMultiativaValorEsperado:
    def __init__(self, df, limite_subida=LIMIT_SUBIDA_MAXIMA):
        self.df = df.copy()
        self.resultados = {}
        self.limite_subida = limite_subida

    def calcular_valor_esperado_por_cenario(self):
        simbolos = self.df['simbolo'].unique()
        info_cenarios = []

        for simbolo in simbolos:
            grupo = self.df[self.df['simbolo'] == simbolo].copy()
            if len(grupo) < 24:  # Filtra para ativos com pelo menos 24 registros
                continue
            grupo = grupo.sort_values('datahora_fechamento')

            grupo['var'] = ((grupo['ultimo_preco'] / grupo['ultimo_preco'].shift(1) - 1) - 0.000575 - 0.001) * 0.85
            grupo = grupo.dropna(subset=['var'])

            grupo['cenario'] = (grupo['var'] > 0).astype(int)  # 1 para positivo, 0 para negativo

            grupo['acumulado_var'] = grupo['var'].cumsum()

            for cenario_valor in grupo['cenario'].unique():
                subgrupo = grupo[grupo['cenario'] == cenario_valor]
                retornos = subgrupo['var']

                p_up = (retornos > 0).mean()
                p_down = 1 - p_up
                mean_up = retornos[retornos > 0].mean() if p_up > 0 else 0
                mean_down = retornos[retornos < 0].mean() if p_down > 0 else 0
                valor_esperado = mean_up * p_up + mean_down * p_down

                info_cenarios.append({
                    'simbolo': simbolo,
                    'cenario': cenario_valor,
                    'valor_esperado': valor_esperado,
                    'num_ocorrencias': len(subgrupo)
                })

            self.resultados[simbolo] = grupo

        df_cenarios = pd.DataFrame(info_cenarios)
        return df_cenarios

    def escolher_ativo_para_investir(self, df_cenarios):
        cenarios_atuais = []
        for simbolo, grupo in self.resultados.items():
            grupo = grupo.sort_values('datahora_fechamento')
            ultimo_registro = grupo.iloc[-1]
            cenario_atual = int(ultimo_registro['var'] > 0)
            acumulado_atual = ultimo_registro['acumulado_var']
            cenarios_atuais.append({
                'simbolo': simbolo,
                'cenario_atual': cenario_atual,
                'acumulado_var': acumulado_atual
            })

        df_cenarios_atuais = pd.DataFrame(cenarios_atuais)
        df_merged = pd.merge(df_cenarios, df_cenarios_atuais, how='inner', on='simbolo')
        df_filtrado = df_merged[df_merged['cenario'] == df_merged['cenario_atual']]
        df_filtrado = df_filtrado[df_filtrado['acumulado_var'] <= self.limite_subida]

        if df_filtrado.empty:
            return None, None, None

        idx_max = df_filtrado['valor_esperado'].idxmax()
        melhor = df_filtrado.loc[idx_max]

        return melhor['simbolo'], melhor['cenario'], melhor['valor_esperado']

    def salvar_dados_ativo(self, simbolo, nome_arquivo='valor_esperado_test.xlsx'):
        if simbolo not in self.resultados:
            print(f"Ativo {simbolo} não encontrado nos resultados para salvar.")
            return
        df_ativo = self.resultados[simbolo].copy()
        df_ativo.to_excel(nome_arquivo, index=False)
        print(f"Dados do ativo {simbolo} salvos em {nome_arquivo}.")


# Executar a estratégia
estrategia = EstrategiaMultiativaValorEsperado(df)
df_cenarios = estrategia.calcular_valor_esperado_por_cenario()

simbolo_escolhido, cenario_escolhido, valor_esperado = estrategia.escolher_ativo_para_investir(df_cenarios)

if simbolo_escolhido is not None:
    grupo = estrategia.resultados[simbolo_escolhido]
    grupo = grupo.sort_values('datahora_fechamento')
    grupo['acumulado_var'] = grupo['var'].cumsum()

    estrategia.salvar_dados_ativo(simbolo_escolhido)

    linha_valor_esperado = np.full(len(grupo), valor_esperado)

    plt.figure(figsize=(14, 7))
    plt.plot(grupo['datahora_fechamento'], grupo['acumulado_var'], label='Acumulado Variação Normal', linewidth=2)
    plt.plot(grupo['datahora_fechamento'], linha_valor_esperado,
             label='Valor Esperado (cenário atual)', color='red', linewidth=2, linestyle=':')
    plt.title(f'Ativo selecionado para investimento: {simbolo_escolhido}\n'
              f'Cenário: {cenario_escolhido} - Valor Esperado: {valor_esperado:.6f}')
    plt.xlabel('Data/Hora de Fechamento')
    plt.ylabel('Variação Acumulada')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("Nenhum ativo atende aos critérios atuais para investimento.")
