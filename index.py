import pandas as pd
import requests
import numpy as np
import matplotlib.pyplot as plt


# Baixar os dados
url = "https://n8n.coinverge.com.br/webhook/precos"
# response = requests.get(url)
data = [
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.25510000",
        "preco_maximo":"0.26730000",
        "preco_minimo":"0.24930000",
        "preco_abertura":"0.26600000",
        "datahora_fechamento":"2025-08-06T14:00:06.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.26640000",
        "preco_maximo":"0.27190000",
        "preco_minimo":"0.24930000",
        "preco_abertura":"0.25610000",
        "datahora_fechamento":"2025-08-06T22:00:00.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.26830000",
        "preco_maximo":"0.27190000",
        "preco_minimo":"0.24930000",
        "preco_abertura":"0.25770000",
        "datahora_fechamento":"2025-08-07T02:00:04.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.26500000",
        "preco_maximo":"0.27190000",
        "preco_minimo":"0.24930000",
        "preco_abertura":"0.25210000",
        "datahora_fechamento":"2025-08-07T06:00:02.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.26590000",
        "preco_maximo":"0.27190000",
        "preco_minimo":"0.25270000",
        "preco_abertura":"0.25580000",
        "datahora_fechamento":"2025-08-07T10:00:02.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.29400000",
        "preco_maximo":"0.29930000",
        "preco_minimo":"0.28220000",
        "preco_abertura":"0.28410000",
        "datahora_fechamento":"2025-08-10T02:00:01.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.29190000",
        "preco_maximo":"0.29930000",
        "preco_minimo":"0.28250000",
        "preco_abertura":"0.28360000",
        "datahora_fechamento":"2025-08-10T06:00:03.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.26040000",
        "preco_maximo":"0.26170000",
        "preco_minimo":"0.24930000",
        "preco_abertura":"0.25820000",
        "datahora_fechamento":"2025-08-06T18:00:06.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.27380000",
        "preco_maximo":"0.27400000",
        "preco_minimo":"0.25270000",
        "preco_abertura":"0.25500000",
        "datahora_fechamento":"2025-08-07T14:00:03.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.27170000",
        "preco_maximo":"0.27620000",
        "preco_minimo":"0.25900000",
        "preco_abertura":"0.26050000",
        "datahora_fechamento":"2025-08-07T18:00:06.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.26880000",
        "preco_maximo":"0.27620000",
        "preco_minimo":"0.26340000",
        "preco_abertura":"0.26640000",
        "datahora_fechamento":"2025-08-07T22:00:05.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.27610000",
        "preco_maximo":"0.27640000",
        "preco_minimo":"0.26340000",
        "preco_abertura":"0.26830000",
        "datahora_fechamento":"2025-08-08T02:00:04.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.28390000",
        "preco_maximo":"0.28930000",
        "preco_minimo":"0.26550000",
        "preco_abertura":"0.26590000",
        "datahora_fechamento":"2025-08-08T10:00:06.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.27980000",
        "preco_maximo":"0.28930000",
        "preco_minimo":"0.26550000",
        "preco_abertura":"0.27370000",
        "datahora_fechamento":"2025-08-08T14:00:04.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.28230000",
        "preco_maximo":"0.28930000",
        "preco_minimo":"0.26550000",
        "preco_abertura":"0.27160000",
        "datahora_fechamento":"2025-08-08T18:00:06.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.28530000",
        "preco_maximo":"0.28930000",
        "preco_minimo":"0.26850000",
        "preco_abertura":"0.26890000",
        "datahora_fechamento":"2025-08-08T22:00:03.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.28420000",
        "preco_maximo":"0.28930000",
        "preco_minimo":"0.27090000",
        "preco_abertura":"0.27610000",
        "datahora_fechamento":"2025-08-09T02:00:05.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.28870000",
        "preco_maximo":"0.29930000",
        "preco_minimo":"0.28690000",
        "preco_abertura":"0.29040000",
        "datahora_fechamento":"2025-08-10T10:00:03.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.28040000",
        "preco_maximo":"0.29930000",
        "preco_minimo":"0.27940000",
        "preco_abertura":"0.29460000",
        "datahora_fechamento":"2025-08-10T14:00:06.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.28510000",
        "preco_maximo":"0.29930000",
        "preco_minimo":"0.27940000",
        "preco_abertura":"0.29250000",
        "datahora_fechamento":"2025-08-10T18:00:06.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.28150000",
        "preco_maximo":"0.29930000",
        "preco_minimo":"0.27940000",
        "preco_abertura":"0.29340000",
        "datahora_fechamento":"2025-08-10T22:00:05.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.27840000",
        "preco_maximo":"0.28980000",
        "preco_minimo":"0.27780000",
        "preco_abertura":"0.28860000",
        "datahora_fechamento":"2025-08-11T10:00:01.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.28370000",
        "preco_maximo":"0.28920000",
        "preco_minimo":"0.27700000",
        "preco_abertura":"0.28060000",
        "datahora_fechamento":"2025-08-09T06:00:05.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.29040000",
        "preco_maximo":"0.29280000",
        "preco_minimo":"0.27700000",
        "preco_abertura":"0.28390000",
        "datahora_fechamento":"2025-08-09T10:00:04.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.29470000",
        "preco_maximo":"0.29690000",
        "preco_minimo":"0.27700000",
        "preco_abertura":"0.27980000",
        "datahora_fechamento":"2025-08-09T14:00:06.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.29250000",
        "preco_maximo":"0.29690000",
        "preco_minimo":"0.27700000",
        "preco_abertura":"0.28230000",
        "datahora_fechamento":"2025-08-09T18:00:02.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.29340000",
        "preco_maximo":"0.29690000",
        "preco_minimo":"0.28220000",
        "preco_abertura":"0.28540000",
        "datahora_fechamento":"2025-08-09T22:00:04.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.28190000",
        "preco_maximo":"0.29760000",
        "preco_minimo":"0.27780000",
        "preco_abertura":"0.29390000",
        "datahora_fechamento":"2025-08-11T02:00:05.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.28360000",
        "preco_maximo":"0.29660000",
        "preco_minimo":"0.27780000",
        "preco_abertura":"0.29190000",
        "datahora_fechamento":"2025-08-11T06:00:04.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.27080000",
        "preco_maximo":"0.28720000",
        "preco_minimo":"0.27040000",
        "preco_abertura":"0.28030000",
        "datahora_fechamento":"2025-08-11T14:00:02.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.26660000",
        "preco_maximo":"0.28720000",
        "preco_minimo":"0.26340000",
        "preco_abertura":"0.28520000",
        "datahora_fechamento":"2025-08-11T18:00:06.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.26590000",
        "preco_maximo":"0.28450000",
        "preco_minimo":"0.26340000",
        "preco_abertura":"0.28150000",
        "datahora_fechamento":"2025-08-11T22:00:05.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.25930000",
        "preco_maximo":"0.28450000",
        "preco_minimo":"0.25680000",
        "preco_abertura":"0.28210000",
        "datahora_fechamento":"2025-08-12T02:00:04.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.26350000",
        "preco_maximo":"0.28360000",
        "preco_minimo":"0.25680000",
        "preco_abertura":"0.28360000",
        "datahora_fechamento":"2025-08-12T06:00:06.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.26320000",
        "preco_maximo":"0.27850000",
        "preco_minimo":"0.25680000",
        "preco_abertura":"0.27850000",
        "datahora_fechamento":"2025-08-12T10:00:04.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.26290000",
        "preco_maximo":"0.27090000",
        "preco_minimo":"0.25680000",
        "preco_abertura":"0.27090000",
        "datahora_fechamento":"2025-08-12T14:00:02.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.26720000",
        "preco_maximo":"0.27060000",
        "preco_minimo":"0.25680000",
        "preco_abertura":"0.26660000",
        "datahora_fechamento":"2025-08-12T18:00:05.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.27370000",
        "preco_maximo":"0.27560000",
        "preco_minimo":"0.25860000",
        "preco_abertura":"0.25920000",
        "datahora_fechamento":"2025-08-13T02:00:05.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.27360000",
        "preco_maximo":"0.27560000",
        "preco_minimo":"0.26020000",
        "preco_abertura":"0.26360000",
        "datahora_fechamento":"2025-08-13T06:00:06.000"
    },
    {
        "simbolo":"1INCHUSDT",
        "ultimo_preco":"0.27870000",
        "preco_maximo":"0.27940000",
        "preco_minimo":"0.26020000",
        "preco_abertura":"0.26290000",
        "datahora_fechamento":"2025-08-13T14:00:06.000"
    }
]

# DataFrame
df = pd.DataFrame(data)
print(df)
# print("DataFrame criado com as seguintes colunas:")
# for col in df.columns:
#     print(f"- {col}")
# print(f"Shape do DataFrame: {df.shape}")

# Conversão numérica
for col in df.columns:
    try:
        df[col] = pd.to_numeric(df[col])
    except Exception:
        pass


# Garantir datetime para ordenar corretamente (remover timezone)
if 'datahora_fechamento' in df.columns:
    try:
        df['datahora_fechamento'] = pd.to_datetime(df['datahora_fechamento']).dt.tz_localize(None)
        print("'datahora_fechamento' convertido para datetime sem timezone.")
    except Exception as e:
        print(f"Erro convertendo 'datahora_fechamento': {e}")


class BacktestStopDinamicoTodos:
    def __init__(self, df):
        self.df = df.copy()
        self.resultados = {}
        print(f"Backtest iniciado com DataFrame contendo {len(self.df)} registros.")

    def simular(self):
        ultimas_variacoes = []
        simbolos = self.df['simbolo'].unique()
        print(f"\n\nSimulação iniciada para {len(simbolos)} ativos.")
        for simbolo in simbolos:
            grupo = self.df[self.df['simbolo'] == simbolo].copy()
            if len(grupo) < 24:  # Funcionando
                continue
            grupo = grupo.sort_values('datahora_fechamento')

            # Cálculo das métricas
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
            
            retornos = grupo['var'].dropna()

            # Probabilidade de subir
            p_up = (retornos > 0).mean()

            # Probabilidade de cair
            p_down = 1 - p_up

            # Média dos retornos positivos
            mean_up = retornos[retornos > 0].mean() if p_up > 0 else 0

            # Média dos retornos negativos
            mean_down = retornos[retornos < 0].mean() if p_down > 0 else 0

            # Valor esperado
            valor_esperado = (mean_up * p_up) + (mean_down * p_down)

            ultimas_variacoes.append({
                'simbolo': simbolo,
                'valor_esperado': valor_esperado
            })


            grupo['acumulado_estrategia'] = grupo['retorno_estrategia'].cumsum()
            grupo['acumulado_var'] = grupo['var'].cumsum()

            self.resultados[simbolo] = grupo

            ultima_var = grupo.iloc[-1]['retorno_estrategia']
            ultimas_variacoes.append({'simbolo': simbolo, 'ultima_var': ultima_var, 'stop': grupo.iloc[-1]['stop']})

        if not ultimas_variacoes:
            print("Nenhum ativo processado na simulação.")
            return None

        df_variacoes = pd.DataFrame(ultimas_variacoes)
        print(f"\n{df_variacoes}")
        melhor_idx = df_variacoes['valor_esperado'].idxmax()

        melhor_ativo = df_variacoes.loc[melhor_idx]
        print(f"Melhor ativo do período: {melhor_ativo['simbolo']} com variação {melhor_ativo['ultima_var']:.6f}")
        return melhor_ativo

    def estrategia_multiativo(self):
        if not self.resultados:
            print("Nenhum resultado disponível para estratégia_multiativo.")
            return None, None

        print("Construindo tabela multiativo...")
        todos = []
        for simbolo, grupo in self.resultados.items():
            df_aux = grupo[['datahora_fechamento', 'retorno_estrategia']].copy()
            df_aux['simbolo'] = simbolo
            todos.append(df_aux)

        df_todos = pd.concat(todos)
        print(f"Dados concatenados para {len(todos)} ativos.")

        df_todos_agg = df_todos.groupby(['datahora_fechamento', 'simbolo'], as_index=False).agg({'retorno_estrategia': 'last'})

        tabela = df_todos_agg.pivot(index='datahora_fechamento', columns='simbolo', values='retorno_estrategia').sort_index()
        print(f"Tabela de retornos multiativo criada com shape {tabela.shape}.")

        escolhas = tabela.idxmax(axis=1)
        retornos_multiativo = tabela.max(axis=1).fillna(0)
        acumulado_multiativo = retornos_multiativo.cumsum()

        info_df = tabela.copy()
        info_df['ativo_recomendado'] = escolhas
        info_df['retorno_recomendado'] = retornos_multiativo
        info_df['confere_recomendacao'] = True

        acumulados_normais = []
        for tempo, ativo in zip(info_df.index, info_df['ativo_recomendado']):
            if ativo in self.resultados:
                df_ativo = self.resultados[ativo]
                linha = df_ativo[df_ativo['datahora_fechamento'] == tempo]
                if not linha.empty:
                    acumulados_normais.append(linha['acumulado_var'].values[0])
                else:
                    acumulados_normais.append(np.nan)
            else:
                acumulados_normais.append(np.nan)
        info_df['acumulado_var_ativo_recomendado'] = acumulados_normais

        info_df_reset = info_df.reset_index()

        primeiro_ativo = info_df_reset.loc[0, 'ativo_recomendado']
        print(f"Primeiro ativo recomendado na série temporal: {primeiro_ativo}")

        arquivo_excel = 'relatorio_detalhado_recomendacoes.xlsx'
        info_df_reset.to_excel(arquivo_excel, index=False)
        print(f"Exportação da tabela multiativo concluída: {arquivo_excel}")

        return info_df_reset, primeiro_ativo

    def calcular_maior_variacao_horaria_quinto(self):
        df = self.df.copy()

        df['datahora_fechamento_hora'] = df['datahora_fechamento'].dt.floor('H')
        df = df.sort_values(['simbolo', 'datahora_fechamento'])

        # Agregar o último preço por ativo por hora
        df_agg = df.groupby(['simbolo', 'datahora_fechamento_hora']).agg({'ultimo_preco': 'last'}).reset_index()

        # Calcular variação percentual horária por ativo (comparando com a hora anterior)
        df_agg['var_horaria'] = df_agg.groupby('simbolo')['ultimo_preco'].pct_change().fillna(0)

        # Função para pegar o quinto maior ativo por hora
        def pegar_quinto_maior(grupo):
            grupo_ordenado = grupo.sort_values('var_horaria', ascending=False).reset_index(drop=True)
            if len(grupo_ordenado) >= 5:
                return grupo_ordenado.loc[4]  # quinto maior
            else:
                return grupo_ordenado.iloc[-1]  # se tem menos que 5, pegar o menor

        resumo = df_agg.groupby('datahora_fechamento_hora').apply(pegar_quinto_maior).reset_index(drop=True)

        # Renomear colunas para clareza
        resumo = resumo.rename(columns={
            'datahora_fechamento_hora': 'datahora_fechamento',
            'var_horaria': 'maior_valor',
            'simbolo': 'ativo_com_maior_variacao'
        })

        # Calcular a variação da próxima hora para o ativo escolhido
        # Montar dataframe auxiliar para facilitar buscas
        df_agg.set_index(['simbolo', 'datahora_fechamento_hora'], inplace=True)
        resumo['proxima_var'] = np.nan  # inicializar coluna

        for idx, row in resumo.iterrows():
            ativo = row['ativo_com_maior_variacao']
            hora_atual = row['datahora_fechamento']
            proxima_hora = hora_atual + pd.Timedelta(hours=1)
            try:
                proxima_var = df_agg.loc[(ativo, proxima_hora), 'var_horaria']
            except KeyError:
                proxima_var = 0  # Se não existe dado para próxima hora assumir 0
            resumo.at[idx, 'proxima_var'] = proxima_var

        # Agora calcular o acumulado da proxima_var considerando acumulado no dia anterior para cada ativo
        # Para isso precisamos do acumulado diário por ativo

        # Primeiro retornamos df_agg ao formato normal para cálculos adicionais
        df_agg.reset_index(inplace=True)

        # Criar coluna de dia para agrupamento diário
        df_agg['dia'] = df_agg['datahora_fechamento_hora'].dt.floor('D')

        # Calcular acumulado diário de var_horaria por ativo
        df_acumulado_diario = df_agg.groupby(['simbolo', 'dia'])['var_horaria'].sum().reset_index()
        df_acumulado_diario = df_acumulado_diario.rename(columns={'var_horaria': 'acumulado_diario'})

        # Agora atribuir para cada linha em resumo o acumulado do dia anterior do ativo
        acumulado_anterior = []
        for idx, row in resumo.iterrows():
            ativo = row['ativo_com_maior_variacao']
            hora = row['datahora_fechamento']
            dia_atual = hora.floor('D')
            dia_anterior = dia_atual - pd.Timedelta(days=1)

            # Procurar acumulado do dia anterior
            filtro = (df_acumulado_diario['simbolo'] == ativo) & (df_acumulado_diario['dia'] == dia_anterior)
            resultado = df_acumulado_diario.loc[filtro, 'acumulado_diario']
            if not resultado.empty:
                acumulado_anterior.append(resultado.values[0])
            else:
                acumulado_anterior.append(0)

        resumo['acumulado_dia_anterior'] = acumulado_anterior

        # Por fim, acumulado da proxima_var somado ao acumulado do dia anterior cumulativamente (cumsum por ativo)
        # Mas o enunciado pede acumulado = proxima_var + acumulado do dia anterior (não cumulativo)

        # resumo['acumulado_proxima_var'] = resumo['proxima_var'] + resumo['acumulado_proxima_var'].cumsum()
        resumo['acumulado_proxima_var'] = (
            resumo.groupby('ativo_com_maior_variacao')['proxima_var']
                .cumsum() + resumo['acumulado_dia_anterior']
        )
        
        # Selecionar colunas finais
        resumo = resumo[['datahora_fechamento', 'maior_valor', 'ativo_com_maior_variacao', 'proxima_var', 'acumulado_proxima_var']]

        print("\nTabela do quinto ativo de maior variação horária com proxima_var e acumulado_proxima_var:")
        print(resumo.head(10))

        return resumo

    def salvar_dados_por_ativo(self, nome_arquivo='dados_por_ativo.xlsx'):
        if not self.resultados:
            print("Nenhum resultado para salvar.")
            return

        print(f"Iniciando salvamento dos dados por ativo no arquivo '{nome_arquivo}'...")
        with pd.ExcelWriter(nome_arquivo) as writer:
            for simbolo, df_ativo in self.resultados.items():
                nome_aba = simbolo[:31]  # Limite de 31 caracteres para nome da aba Excel
                df_ativo.to_excel(writer, sheet_name=nome_aba, index=False)
        print(f"Dados de todos os ativos salvos com sucesso em: {nome_arquivo}")


# === Execução do Backtest ===

backtest = BacktestStopDinamicoTodos(df)
melhor_do_periodo = backtest.simular()

print("\nResultado final:")
if melhor_do_periodo is not None:
    simbolo_escolhido = melhor_do_periodo['simbolo']
    variacao_escolhida = melhor_do_periodo['ultima_var']

    backtest.salvar_dados_por_ativo('detalhes_ativos.xlsx')

    print("################################")
    tabela_quinto_maior = backtest.calcular_maior_variacao_horaria_quinto()

    tabela_info, primeiro_ativo = backtest.estrategia_multiativo()

    print(f"\nPrimeiro ativo recomendado: {primeiro_ativo}")
    print("\nTabela detalhada das recomendações e variações (primeiras 10 linhas):")

    # Obter df_ativo para o ativo escolhido
    df_ativo = backtest.resultados[simbolo_escolhido]


    plt.figure(figsize=(14, 7))
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
