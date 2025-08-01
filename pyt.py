import pandas as pd
import requests

# Baixar dados
url = "https://n8n.coinverge.com.br/webhook/precos"
response = requests.get(url)
data = response.json()
df = pd.DataFrame(data)

# Converter colunas para tipo numérico quando possível
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except Exception:
        pass

# Garantir datetime para manipulação e ordenação
if 'datahora_fechamento' in df.columns:
    df['datahora_fechamento'] = pd.to_datetime(df['datahora_fechamento']).dt.tz_localize(None) 
    # arredonda datahora_fechamento para hora mais próxima
    df['datahora_fechamento'] = df['datahora_fechamento'].dt.floor('H')
    # soma os valores de data repetidos por simbolo
    print(df)

# Calcular variação por ativo
dfs = []
for simbolo in df['simbolo'].unique():
    grupo = df[df['simbolo'] == simbolo].sort_values('datahora_fechamento').copy()
    grupo['var'] = grupo['ultimo_preco'] / grupo['ultimo_preco'].shift(1) - 1
    # Pega só colunas necessárias
    dfs.append(grupo[['datahora_fechamento', 'simbolo', 'var']])

# Concatenar resultados
df_var = pd.concat(dfs)

# Gera grid completo com todas datahora_fechamento possíveis
todos_horarios = pd.DataFrame({'datahora_fechamento': df_var['datahora_fechamento'].sort_values().unique()})

# Para cada ativo, faz merge com todos_horarios para não perder nenhuma data/hora
ativos = df_var['simbolo'].unique()
tabelas_ativos = []

tabela_var = pd.pivot_table(
    df_var,
    values='var',
    index='datahora_fechamento',
    columns='simbolo'
).reset_index()

# Para garantir que todas as datas estejam presentes (todos_horarios),
# você pode fazer um merge com a tabela de horários completa:

tabela_var = todos_horarios.merge(tabela_var, on='datahora_fechamento', how='left')

# Remove índice para exportar
tabela_var_reset = tabela_var.reset_index()

# Opcional: preenche vazios com '' ou 0 (cuidado, muda sentido estatístico)
# tabela_var_reset = tabela_var_reset.fillna('')

# Exporta
arquivo_excel = 'variacao_ativos_grid_universal.xlsx'
tabela_var_reset.to_excel(arquivo_excel, index=False)
print(f'Tabela de variação por ativo com grid universal salva em {arquivo_excel}')
