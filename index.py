import pandas as pd
import requests

# Baixar os dados do link
url = "https://n8n.coinverge.com.br/webhook/precos"
response = requests.get(url)
data = response.json()

# Converter para DataFrame
df = pd.DataFrame(data)

# Exemplo de conversão de tipos de dados
# Ajuste conforme os tipos reais dos dados
for col in df.columns:
    # Tenta converter para numérico, se possível
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Exibe os tipos de dados convertidos
print(df.dtypes)

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display  # Para display em Jupyter

class BacktestStopDinamico:
    def __init__(self, df):
        self.df = df.copy()
        self.resultados = {}

    def simular(self):
        for simbolo, grupo in self.df.groupby('simbolo'):
            if simbolo != '1INCHUSDT':
                continue
            if len(grupo) < 24:
                continue
            
            grupo = grupo.sort_values('datahora_fechamento').copy()
            grupo['var'] = ((grupo['ultimo_preco'] / grupo['ultimo_preco'].shift(1) - 1) - 0.000575 - 0.001) * 0.85
            grupo['max_desceu'] = (grupo['preco_minimo'] / grupo['ultimo_preco'].shift(1) - 1)
            grupo['stop'] = np.where(grupo['var'].shift(1) < 0, 0, grupo['max_desceu'].shift(1))
            # grupo['stop'] = np.nan  # Primeira linha: NaN
            # if len(grupo) > 1:
            #     grupo.iloc[1, grupo.columns.get_loc('stop')] = 0  # Segunda linha: 0
            # if len(grupo) > 2:
            #     grupo.iloc[2:, grupo.columns.get_loc('stop')] = np.where(
            #         grupo['var'].shift(1).iloc[2:] < 0,
            #         0,
            #         grupo['max_desceu'].shift(1).iloc[2:]
            #     )
            grupo['retorno_estrategia'] = np.nan
            grupo['acumulado_estrategia'] = np.nan
            grupo['variacao_acumulado_max_desceu'] = np.nan
            

            prev_stop = None
            prev_acumulado = 1
            max_acumulado = 1
            stop_vals = []
            retorno_estrategia = []
            acumulado = []
            variacao_max_desceu = []

            for i in range(len(grupo)):
                if i == 0:
                    # Primeira linha: campos vazios
                    stop_vals.append(np.nan)
                    retorno_estrategia.append(np.nan)
                    acumulado.append(np.nan)
                    variacao_max_desceu.append(np.nan)
                    continue

                fechamento = grupo.iloc[i]['ultimo_preco']
                minimo = grupo.iloc[i]['preco_minimo']
                abertura = grupo.iloc[i]['preco_abertura']

                prev_fechamento = grupo.iloc[i-1]['ultimo_preco']
                prev_minimo = grupo.iloc[i-1]['preco_minimo']
                prev_stop = (prev_fechamento - prev_minimo) / prev_fechamento # ISSO ESTA ERRADO
                stop_vals.append(prev_stop)

                stop_preco = abertura * (1 - prev_stop)
                if minimo <= stop_preco:
                    retorno = (stop_preco - abertura) / abertura
                else:
                    retorno = (fechamento - abertura) / abertura

                retorno_estrategia.append(retorno)
                prev_acumulado *= (1 + retorno)
                acumulado.append(prev_acumulado)
                max_acumulado = max(max_acumulado, prev_acumulado)
                variacao_max_desceu.append((prev_acumulado - max_acumulado) / max_acumulado)

            grupo['stop'] = stop_vals
            grupo['retorno_estrategia'] = retorno_estrategia
            grupo['acumulado_estrategia'] = acumulado
            grupo['variacao_acumulado_max_desceu'] = variacao_max_desceu

            self.resultados[simbolo] = grupo
            grupo.to_excel(f'excel/{simbolo}.xlsx', index=False)  # Salvar resultados em Excel

    def plot_tabelas(self):
        # Limitar o número de ativos plotados para evitar sobrecarga de desempenho (ex: primeiros 5)
        simbolos_para_plotar = list(self.resultados.keys())  # Ajuste o número conforme necessário
        i=0
        for simbolo in self.resultados.keys():
            if i < 5:  # Limitar a plotar apenas os 2 primeiros
                i=i+1
                resultado = self.resultados[simbolo]
                # print(f"Tabela de resultados para {simbolo}:")
                print(f"Tabela de resultados para {simbolo}:")
                # display(resultado[['datahora_fechamento', 'preco_abertura', 'ultimo_preco', 'preco_minimo', 'stop', 'retorno_estrategia', 'acumulado_estrategia', 'variacao_acumulado_max_desceu']].head(20))
                
                # Plotar gráfico da variação do stop por ativo, mas apenas se for um dos selecionados e com poucos registros (ex: head(50))
                # if simbolo in simbolos_para_plotar:
                #     df_plot = resultado.head(50)  # Limitar a 50 registros para desempenho
                #     plt.figure(figsize=(10, 6))
                #     plt.plot(df_plot['datahora_fechamento'], df_plot['acumulado_estrategia'], marker='o', linestyle='-', color='b')
                #     plt.title(f'Variação do Stop para {simbolo}')
                #     plt.xlabel('Data/Hora de Fechamento')
                #     plt.ylabel('Valor do Stop')
                #     plt.xticks(rotation=45)
                #     plt.grid(True)
                #     plt.tight_layout()
                #     plt.show()

# Exemplo de uso:
backtest = BacktestStopDinamico(df)
backtest.simular()
backtest.plot_tabelas()
