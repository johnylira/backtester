import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display  # Para Jupyter

# Baixar os dados do link
url = "https://n8n.coinverge.com.br/webhook/precos"
response = requests.get(url)
data = response.json()

# Converter para DataFrame
df = pd.DataFrame(data)

# Conversão segura de tipos
for col in df.columns:
    # Tenta converter para numérico, se possível
    df[col] = pd.to_numeric(df[col], errors='ignore')

# Exibir tipos de dados
print(df.dtypes)

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

            grupo['stop'] = np.where((grupo['var'].shift(1).fillna(0) < 0), 0, grupo['max_desceu'].shift(1).fillna(0))
            grupo['retorno_estrategia'] = np.where(grupo['max_desceu'] < grupo['stop'], (grupo['stop'] - 0.000576 - 0.001) * 0.85, grupo['var'])
            grupo['acumulado_estrategia'] = grupo['retorno_estrategia'].cumsum()

            #=IF(H3<I3;(I3-0,0575%-0,1%)*0,85;F3)

            self.resultados[simbolo] = grupo
            grupo.to_excel(f'excel/{simbolo}.xlsx', index=False)

    # def plot_tabelas(self):
    #     simbolos_para_plotar = list(self.resultados.keys())
    #     i = 0
    #     for simbolo in self.resultados.keys():
    #         if i < 5:
    #             i += 1
    #             resultado = self.resultados[simbolo]
    #             print(f"Tabela de resultados para {simbolo}:")
    #             display(resultado[[
    #                 'datahora_fechamento', 'preco_abertura', 'ultimo_preco', 'preco_minimo',
    #                 'stop', 'retorno_estrategia', 'acumulado_estrategia', 'variacao_acumulado_max_desceu'
    #             ]].head(20))

# Executar
backtest = BacktestStopDinamico(df)
backtest.simular()
# backtest.plot_tabelas()
