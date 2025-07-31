import pandas as pd
import requests

# Baixar os dados
url = "https://n8n.coinverge.com.br/webhook/precos"
response = requests.get(url)
data = response.json()

# Criar DataFrame
df = pd.DataFrame(data)

# Converter colunas para tipo numérico quando possível
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except Exception:
        pass

# Garantir datetime para facilitar ordenação e manipulação
if 'datahora_fechamento' in df.columns:
    df['datahora_fechamento'] = pd.to_datetime(df['datahora_fechamento']).dt.tz_localize(None)

# Calcular variação de preço por ativo
# var = (preço_atual / preço_anterior) - 1
dfs = []
for simbolo in df['simbolo'].unique():
    grupo = df[df['simbolo'] == simbolo].sort_values('datahora_fechamento').copy()
    grupo['var'] = grupo['ultimo_preco'] / grupo['ultimo_preco'].shift(1) - 1
    dfs.append(grupo[['datahora_fechamento', 'simbolo', 'var']])

# Concatenar resultados
df_var = pd.concat(dfs)

# Agrupar por datahora_fechamento e simbolo para garantir unicidade, usando média das variações
df_var_agg = df_var.groupby(['datahora_fechamento', 'simbolo'], as_index=False).agg({'var': 'mean'})

# Pivotar para formar a tabela com ativos nas colunas e variações nas linhas
tabela_var = df_var_agg.pivot(index='datahora_fechamento', columns='simbolo', values='var')

# Exportar para Excel
arquivo_excel = 'variacao_por_ativo.xlsx'
tabela_var.to_excel(arquivo_excel)

print(f'Tabela de variação por ativo salva em {arquivo_excel}')
