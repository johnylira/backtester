import pandas as pd
import requests

# Baixar dados da API
url = "https://n8n.coinverge.com.br/webhook/precos"
response = requests.get(url)
data = response.json()

df = pd.DataFrame(data)

# Defina aqui o símbolo desejado exatamente como aparece na API
simbolo_desejado = "1INCHUSDT"

# Filtrar pelo símbolo
df_simbolo = df[df['simbolo'] == simbolo_desejado].copy()

# Converter para datetime
df_simbolo['datahora_fechamento'] = pd.to_datetime(
    df_simbolo['datahora_fechamento']
).dt.tz_localize(None)

# Manter apenas as colunas desejadas
colunas_desejadas = [
    'simbolo',
    'ultimo_preco',
    'preco_maximo',
    'preco_minimo',
    'preco_abertura',
    'datahora_fechamento'
]
df_final = df_simbolo[colunas_desejadas]

# Salvar como JSON
df_final.to_json("test_excel.json", orient="records", date_format="iso", indent=4, force_ascii=False)

print("Arquivo 'test_excel.json' criado com sucesso!")
print(df_final)
