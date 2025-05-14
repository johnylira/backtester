import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datetime import datetime

class Simulador:
    def __init__(self, params=None) -> None:
        print('===============     Simulador.__init__()     =================')
        self.params = params
        self.preditores_treino, self.preditores_teste, self.alvo_treino, self.alvo_teste = (None, None, None, None)
        self.sigla_ativo = params['sigla_ativo'] if 'sigla_ativo' in params else 'BTC-USD' #sigla_ativo
        self.data_inicial = params['data_inicial'] if 'data_inicial' in params else '2023-01-01' #data_inicial
        self.data_final = params['data_final'] if 'data_final' in params else '2024-11-01' #data_final
        self.modelo = params['modelo'] if 'modelo' in params else DecisionTreeClassifier(random_state=42)
        self.DEBUG = params['DEBUG'] if 'DEBUG' in params else False
        self.janela_treino = params['janela_treino'] if 'janela_treino' in params else 365
        self.preditores = params['preditores'] if 'preditores' in params else [
            'rsi_14', 'mm_exp_21', 'razao_amplitude', 'max_div_min', 'ret_dia_anterior'
        ]
        self.proporcao_teste = params['proporcao_teste'] if 'proporcao_teste' in params else 0.25
        self.stop_loss = params['stop_loss'] if 'stop_loss' in params else {'tipo': 'fixo', 'valor': 0.05}
        self.intervalo = params['intervalo'] if 'intervalo' in params else '1d'
        self.df = None
    
    def getRetornoMedioHorario(self) -> None:
        print('===============     Simulador.getRetornoMedioHorario()     =================')
        a = sim.df.copy()
        a=a.reset_index()[['Close','Datetime']]
        a['hora_inicio'] = a['Datetime'].dt.hour
        a['retorno_abertura_fechamento'] = a['Close'].pct_change()
        a=a.dropna()
        print(a[['retorno_abertura_fechamento','hora_inicio']].groupby('hora_inicio').mean())
    
    def getPrecos(self) -> None:
        print('===============     Simulador.getPrecos()     =================')
        self.df = yf.download(self.sigla_ativo, start=self.data_inicial, end=self.data_final, interval=self.intervalo)
        self.df['data'] = self.df.index
        self.df = self.df.copy()  # Garante que self.df é uma cópia independente
        print('data inicial:', self.df.index[0], 'data final:', self.df.index[-1], 'linhas na base:', len(self.df))
    
    def addIndicadores(self) -> None:
        print('===============     Simulador.addIndicadores()     =================')
        # self.df['rsi_7'] = RSIIndicator(self.df['Close'], window=7).rsi()
        # self.df['rsi_30'] = RSIIndicator(self.df['Close'], window=30).rsi()
        # self.df['mm_exp_7'] = EMAIndicator(self.df['Close'], window=7).ema_indicator() / self.df['Close']
        # self.df['mm_exp_30'] = EMAIndicator(self.df['Close'], window=30).ema_indicator() / self.df['Close']

        # self.df['razao_mm_exp_7_30'] = self.df['mm_exp_7'] / self.df['mm_exp_30']

        # self.df['razao_amplitude'] = (self.df['Close'] - self.df['Low']) / (self.df['High'] - self.df['Low'])
        # self.df['max_div_min'] = (self.df['High'] / self.df['Low']) - 1
        # self.df['ret_dia_anterior'] = self.df['Close'].pct_change()
        # self.df['tendencia'] = list(range(len(self.df)))

        self.df['open_0'] = np.log(self.df['Open'])
        self.df['open_1'] = np.log(self.df['open_0'].shift(1))
        self.df['open_2'] = np.log(self.df['open_0'].shift(2))
        self.df['open_3'] = np.log(self.df['open_0'].shift(3))
        self.df['open_4'] = np.log(self.df['open_0'].shift(4))
        self.df['open_5'] = np.log(self.df['open_0'].shift(5))
        self.df['high_0'] = np.log(self.df['High'])
        self.df['high_1'] = np.log(self.df['high_0'].shift(1))
        self.df['high_2'] = np.log(self.df['high_0'].shift(2))
        self.df['high_3'] = np.log(self.df['high_0'].shift(3))
        self.df['high_4'] = np.log(self.df['high_0'].shift(4))
        self.df['high_5'] = np.log(self.df['high_0'].shift(5))
        self.df['low_0'] = np.log(self.df['Low'])
        self.df['low_1'] = np.log(self.df['low_0'].shift(1))
        self.df['low_2'] = np.log(self.df['low_0'].shift(2))
        self.df['low_3'] = np.log(self.df['low_0'].shift(3))
        self.df['low_4'] = np.log(self.df['low_0'].shift(4))
        self.df['low_5'] = np.log(self.df['low_0'].shift(5))
        self.df['close_0'] = np.log(self.df['Close'])
        self.df['close_1'] = np.log(self.df['close_0'].shift(1))
        self.df['close_2'] = np.log(self.df['close_0'].shift(2))
        self.df['close_3'] = np.log(self.df['close_0'].shift(3))
        self.df['close_4'] = np.log(self.df['close_0'].shift(4))
        self.df['close_5'] = np.log(self.df['close_0'].shift(5))

        # # Selecionar apenas as colunas que deseja normalizar
        # cols_to_normalize = [
        #     'open_0', 'open_1', 'open_2', 'open_3', 'open_4', 'open_5',
        #     'high_0', 'high_1', 'high_2', 'high_3', 'high_4', 'high_5',
        #     'low_0', 'low_1', 'low_2', 'low_3', 'low_4', 'low_5',
        #     'close_0', 'close_1', 'close_2', 'close_3', 'close_4', 'close_5'
        # ]

        # # Normalizar cada linha para que seus valores fiquem entre 0 e 1
        # self.df[cols_to_normalize] = self.df[cols_to_normalize].apply(
        #     lambda row: (row - row.min()) / (row.max() - row.min()) if row.max() != row.min() else np.zeros(len(row)),
        #     axis=1
        # )

        # Selecionar as colunas a serem ajustadas
        cols_to_adjust = [
            'open_0', 'open_1', 'open_2', 'open_3', 'open_4', 'open_5',
            'high_0', 'high_1', 'high_2', 'high_3', 'high_4', 'high_5',
            'low_0', 'low_1', 'low_2', 'low_3', 'low_4', 'low_5',
            'close_1', 'close_2', 'close_3', 'close_4', 'close_5'
        ]

        # Subtrair 'close_0' de cada uma das colunas selecionadas
        for col in cols_to_adjust:
            self.df[col] = self.df[col] - self.df['close_0']

        print('ANTES:',len(self.df), 'linhas na base:', 'de', self.df.index[0], 'até', self.df.index[-1])
        self.df = self.df.copy().dropna()
        print('DEPOIS:',len(self.df), 'linhas na base:', 'de', self.df.index[0], 'até', self.df.index[-1])
        print('===========================================================')
        print(self.df.head(1).T)
    
    def addAlvo(self) -> None:
        print('===============     Simulador.addAlvo()     =================')
        self.df['alvo'] = (self.df['Close'].shift(-1) > self.df['Close']).astype(int)
        # print('ANTES:',len(self.df), 'linhas na base:', 'de', self.df.index[0], 'até', self.df.index[-1])
        # self.df = self.df.copy().dropna()
        # print('DEPOIS:',len(self.df), 'linhas na base:', 'de', self.df.index[0], 'até', self.df.index[-1])
        # print('===========================================================')

    def runDivisaoTreinoTeste(self) -> tuple:
        print('===============     Simulador.runDivisaoTreinoTeste()     =================')
        X = self.df[self.preditores]
        y = self.df['alvo']
        return train_test_split(X, y, test_size=self.proporcao_teste, random_state=42, shuffle=False)
    
    def runTreinarModelo(self, modelo, X_train, y_train) -> None:
        modelo.fit(X_train, y_train)
    
    def addSinal(self, X_train, X_test, so_compra=False) -> None:
        print('===============     Simulador.addSinal()     =================')
        self.df.loc[:, 'sinal'] = pd.Series(np.nan, index=self.df.index)
        self.df.loc[X_train.index, 'Close'] = pd.Series(np.nan, index=X_train.index)
        if self.params['so_compra']==True:
            self.df.loc[X_test.index, 'sinal'] = self.modelo.predict(X_test)
        else:
            self.df.loc[X_test.index, 'sinal'] = self.modelo.predict(X_test)
            self.df.loc[X_test.index, 'sinal'] = self.df.loc[X_test.index, 'sinal'].map({1: 1, 0: -1})
    
    def addRetornos(self) -> None:
        print('===============     Simulador.addRetornos()     =================')
        self.df.loc[self.df['sinal'].notna(),'retornos_estrategia'] = \
            self.df.loc[self.df['sinal'].notna(),'sinal'].shift(1) * self.df.loc[self.df['sinal'].notna(),'Close'].pct_change()
        self.df['retornos_acumulados_estrategia'] = (1 + self.df['retornos_estrategia']).cumprod() - 1

        self.df.loc[self.df['sinal'].notna(),'retornos_compra_e_segura'] = \
            self.df.loc[self.df['sinal'].notna(),'Close'].pct_change()
        self.df['retornos_acumulados_compra_e_segura'] = (1 + self.df['retornos_compra_e_segura']).cumprod() - 1

        if params['stop_loss']['tipo'] == 'fixo':
            self.df.loc[self.df['sinal'].notna(), 'retornos_compra_e_segura_stop'] = \
                np.where(self.df.loc[self.df['sinal'].notna(), 'variacao_close_min'] < -params['stop_loss']['valor'],
                        -params['stop_loss']['valor'],
                        self.df.loc[self.df['sinal'].notna(), 'retornos_compra_e_segura'])
            self.df['retornos_acumulados_compra_e_segura_stop'] = (1 + self.df['retornos_compra_e_segura_stop']).cumprod() - 1

            self.df.loc[self.df['sinal'].notna(), 'retornos_estrategia_stop'] = \
                np.where(self.df.loc[self.df['sinal'].notna(), 'variacao_close_min'] < -params['stop_loss']['valor'],
                        -params['stop_loss']['valor'],
                        self.df.loc[self.df['sinal'].notna(), 'retornos_estrategia'])
            self.df['retornos_acumulados_estrategia_stop'] = (1 + self.df['retornos_estrategia_stop']).cumprod() - 1

        elif params['stop_loss']['tipo'] == 'dinamico':
            self.df.loc[self.df['sinal'].notna(), 'retornos_compra_e_segura_stop'] = \
                np.where(self.df.loc[self.df['sinal'].notna(), 'variacao_close_min'] < self.df.loc[self.df['sinal'].notna(), 'variacao_close_min'].shift(1) - params['stop_loss']['valor'],
                        self.df.loc[self.df['sinal'].notna(), 'variacao_close_min'].shift(1),
                        self.df.loc[self.df['sinal'].notna(), 'retornos_compra_e_segura'])
            self.df['retornos_acumulados_compra_e_segura_stop'] = (1 + self.df['retornos_compra_e_segura_stop']).cumprod() - 1

            self.df.loc[self.df['sinal'].notna(), 'retornos_estrategia_stop'] = \
                np.where(self.df.loc[self.df['sinal'].notna(), 'variacao_close_min'] < self.df.loc[self.df['sinal'].notna(), 'variacao_close_min'].shift(1) - params['stop_loss']['valor'],
                        self.df.loc[self.df['sinal'].notna(), 'variacao_close_min'].shift(1),
                        self.df.loc[self.df['sinal'].notna(), 'retornos_estrategia'])
            self.df['retornos_acumulados_estrategia_stop'] = (1 + self.df['retornos_estrategia_stop']).cumprod() - 1
    
    def runGraficos(self) -> None:
        print('===============     Simulador.runGraficos()     =================')
        self.df.set_index('data', inplace=True)
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['retornos_acumulados_estrategia'], label='Estratégia', linestyle='--')
        plt.plot(self.df['retornos_acumulados_compra_e_segura'], label='Buy & Hold', linestyle=':')
        if params['stop_loss']['tipo'] == 'fixo':
            plt.plot(self.df['retornos_acumulados_estrategia_stop'], label=f'Estratégia com Stop Loss de {params["stop_loss"]["valor"]}', linestyle='-.')
            plt.plot(self.df['retornos_acumulados_compra_e_segura_stop'], label=f'Buy & Hold com Stop Loss de {params["stop_loss"]["valor"]}', linestyle='-')
        elif params['stop_loss']['tipo'] == 'dinamico':
            plt.plot(self.df['retornos_acumulados_estrategia_stop'], label=f'Estratégia com Stop Loss dinâmico de {params["stop_loss"]["valor"]}', linestyle='-.')
            plt.plot(self.df['retornos_acumulados_compra_e_segura_stop'], label=f'Buy & Hold com Stop Loss dinâmico de {params["stop_loss"]["valor"]}', linestyle='-')
        plt.title('Comparação de Retornos Acumulados')
        plt.xlabel('Data')
        plt.ylabel('Retorno Acumulado')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def getAcuracia(self, previsto, real) -> float:
        print('===============     Simulador.getAcuracia()     =================')
        accuracy = accuracy_score(previsto, real)
        print(f"Acurácia no conjunto de teste: {accuracy:.2%}")
        return accuracy
    
    def getRecomendacoes(self) -> None:
        print('===============     Simulador.getRecomendacoes()     =================')
        print('Recomendações:')
        if self.df[['sinal']].values[-1][0] == 1:
            print(self.df['data'].values[-1],'COMPRAR', self.sigla_ativo, 'no preço', self.df['Close'].values[-1])
        else:
            print(self.df['data'].values[-1],'VENDER', self.sigla_ativo, 'no preço', self.df['Close'].values[-1])
    
    def addVariacaoCloseMin(self) -> None:
        print('===============     Simulador.addVariacaoCloseMin()     =================')
        self.df['variacao_close_min'] = self.df['Low']/self.df['Close'].shift(1) - 1

    def runSimulacaoInterativa(self) -> None:
        print('===============     Simulador.runSimulacaoInterativa()     =================')
        """
        Executa o forward backtesting: treina o modelo em cada dia e testa no próximo dia.
        """
        self.getPrecos()
        self.addIndicadores()
        self.addAlvo()
        resultados = []
        for i in range(len(self.df) - self.janela_treino):
            modelo = self.modelo # Cria uma cópia do modelo para cada iteração
            treino = self.df.iloc[i:(i + self.janela_treino)]
            teste = self.df.iloc[(i + self.janela_treino):(i + self.janela_treino + 1)]
            print('Testando o modelo em:', teste.index[0], 'interação', i, 'linha', i + self.janela_treino)
            
            X_train = treino[self.preditores]
            y_train = treino['alvo']
            
            X_test = teste[self.preditores]
            y_test = teste['alvo']
            
            if X_test.isnull().values.any():
                print(f"Valores ausentes encontrados em X_test na data {teste.index[0]}. Pulando previsão.")
                continue  # Ignora iteração se houver NaN

            # Treinando o modelo com os dados de treino até o dia atual
            self.runTreinarModelo(modelo, X_train, y_train)

            if self.params['so_compra']==True:
                previsao = self.modelo.predict(X_test)[0]
            else:
                if self.modelo.predict(X_test)[0] == 1:
                    previsao = 1
                elif self.modelo.predict(X_test)[0] == 0:
                    previsao = -1
                else:
                    previsao = self.modelo.predict(X_test)[0]

            resultados.append({
                'data': teste.index[0],
                'sinal': previsao,
                'real': y_test.iloc[0]
            })

            modelo = None  # Limpa o modelo para o próximo dia
        
        # Convertendo resultados para DataFrame
        df_sim_interativa = pd.DataFrame(resultados)
        
        # Une o DataFrame com os resultados ao DataFrame original
        self.df = pd.merge( self.df, 
                            df_sim_interativa, 
                            how='left', 
                            left_on='data',
                            right_on='data'
                            )
        self.addVariacaoCloseMin()
        self.addRetornos()
        if self.DEBUG == True:
            df_sim_interativa.to_excel('df_sim_interativa.xlsx')
            self.df.to_excel('DEBUG.xlsx')

    def runSimulacao(self) -> None:
        print('===============     Simulador.runSimulacao()     =================')
        self.getPrecos()
        self.addIndicadores()
        self.addAlvo()
        self.preditores_treino, self.preditores_teste, self.alvo_treino, self.alvo_teste = \
            self.runDivisaoTreinoTeste()
        self.runTreinarModelo(modelo = self.modelo, X_train=self.preditores_treino, y_train=self.alvo_treino)
        self.addSinal(self.preditores_treino, self.preditores_teste, so_compra=False)
        # self.addCloseToMin()
        self.addRetornos()

    def getResultados(self) -> None:
        print('===============     Simulador.getResultados()     =================')
        df_acuracia = self.df[['sinal', 'alvo']].dropna().copy()
        self.getAcuracia(df_acuracia['sinal'], df_acuracia['alvo'])
        # cabecalho = pd.DataFrame(cabecalho_parametros, index=[0]) # coloca só na 1ª linha
        # resultado = pd.concat([cabecalho, self.df.reset_index()], axis=1)
        cabecalho = pd.DataFrame([self.params] * len(self.df), index=self.df.index)
        resultado = pd.concat([cabecalho, self.df], axis=1)
        try:
            resultado.to_excel(f'simulacao_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx')
        except:
            print('Erro ao salvar o arquivo xlsx de simulação')
        print('arquivo', f'simulacao_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx', 'salvo com sucesso')
        self.getRecomendacoes()

    def getOportunidade(self, plotar_grafico = False):
        # Faz uma cópia do DataFrame para evitar alterações não intencionais no original
        df = self.df.copy()

        # Calcula a variação percentual (pct_change) e remove valores NaN
        close = df['Close']
        pct_change = close.pct_change().dropna()
        minimo = df['Low']
        volatilidade = (close.shift(1) / minimo - 1).iloc[1:]

        # Garante que os índices estejam alinhados entre `volatilidade` e `pct_change`
        aligned_indices = pct_change.index.intersection(volatilidade.index)
        pct_change = pct_change.loc[aligned_indices]
        volatilidade = volatilidade.loc[aligned_indices]

        oportunidade = volatilidade + pct_change

        if plotar_grafico == True:
            # Adiciona cumsum para cada coluna do DataFrame no gráfico
            for column in oportunidade.columns:
                plt.plot(oportunidade[column].cumsum(), label=column)
            plt.legend()
            plt.show()
            return oportunidade
    
    def getDescritivaMarkov(self, n_classes = 2, periodo_variacao = 1,plotar_grafico = False) -> None:
        print('===============     Simulador.getDescritivaMarkov()     =================')


        df = self.df.copy()
        close = df['Close']
        pct_change = close.pct_change(periodo_variacao).dropna().copy()
        print(pct_change)
        # Criar as colunas SS, SN, NN, NS para cada ativo
        def classificacao(row, ativo):
            
            try:
                if n_classes == 2:
                    return str(int(row[f'{ativo}_bin'])) + str(int(row[f'{ativo}_bin_t-1']))
                if n_classes == 3:
                    return str(int(row[f'{ativo}_bin'])) + str(int(row[f'{ativo}_bin_t-1'])) + str(int(row[f'{ativo}_bin_t-2']))
            except:
                return np.nan
            # if row[f'{ativo}_bin'] == 1 and row[f'{ativo}_bin_t-1'] == 1:
            #     return 'SS'
            # elif row[f'{ativo}_bin'] == 1 and row[f'{ativo}_bin_t-1'] == 0:
            #     return 'SN'
            # elif row[f'{ativo}_bin'] == 0 and row[f'{ativo}_bin_t-1'] == 1:
            #     return 'NS'
            # else:
            #     return 'NN'

        retornos, probabilidades, esperancas = {}, {}, {}

        for col in pct_change.columns:
            # Passo 1: Criar as colunas de retornos anteriores t-1 e t-2 para cada ativo
            pct_change[f'{col}_t-1'] = pct_change[col].shift(periodo_variacao)
            if n_classes == 3:
                pct_change[f'{col}_t-2'] = pct_change[col].shift(2*periodo_variacao)

            # Passo 2: Transformar os retornos em binários (0 = desceu, 1 = subiu)
            pct_change[f'{col}_bin'] = (pct_change[col] > 0).astype(int)
            pct_change[f'{col}_bin_t-1'] = (pct_change[f'{col}_t-1'] > 0).astype(int)
            if n_classes == 3:
                pct_change[f'{col}_bin_t-2'] = (pct_change[f'{col}_t-2'] > 0).astype(int)

            # Passo 3: Criar as colunas SS, SN, NN, NS para cada ativo
            pct_change[f'{col}_classificacao'] = pct_change.apply(lambda row: classificacao(row, col), axis=1)

            # Passo 4: Calcular os retornos futuros t+1
            pct_change[f'{col}_retorno_t+1'] = pct_change[col].shift(-periodo_variacao)
            pct_change[f'{col}_resultado_t+1'] = (pct_change[f'{col}_retorno_t+1'] > 0).astype(int)

            # Passo 5: Calcular as somas dos retornos por classificação
            retornos[col] = pct_change.groupby(f'{col}_classificacao')[f'{col}_retorno_t+1'].mean()

            # Passo 6: Calcular as probabilidades condicionais
            probabilidades[f'{col}_prob'] = pct_change.groupby(f'{col}_classificacao')[f'{col}_resultado_t+1'].mean()

            # Passo 7: Calcular os valores esperados
            esperancas[f'{col}_esperado'] = probabilidades[f'{col}_prob'] * retornos[col]

            # Passo 8: Gerar a coluna com o retorno do ativo que tem maior esperança na classificação
            pct_change[f'{col}_retorno_esperado'] = pct_change[f'{col}_classificacao'].map(esperancas[f'{col}_esperado'])

        # Passo 9: Cria a coluna com o nome da coluna de retornos da estrategia com base no maior retorno esperado
        pct_change['retorno_esperado_escolhido'] = pct_change\
            [[f'{ativo}_retorno_esperado' for ativo in self.sigla_ativo]].idxmax(axis=1)
            
        # Passo 10: Cria coluna com os respectivos retornos da estrategia
        pct_change['retorno_estrategia'] = (pct_change.apply(lambda row: row[row['retorno_esperado_escolhido'].split('_')[0]+'_retorno_t+1'], axis=1)/periodo_variacao).cumsum()

        # Gerar um DataFrame final com as somas dos retornos por classificação para todos os ativos
        retornos, probabilidades, esperancas = pd.DataFrame(retornos), pd.DataFrame(probabilidades), pd.DataFrame(esperancas)
        if plotar_grafico == True:
            plt.plot(pct_change['retorno_estrategia'])
            plt.show()
        print('===============     Simulador.getResultados()     =================')
        print('Retornos esperados: ')
        x = pct_change[[f'{ativo}_retorno_esperado' for ativo in self.sigla_ativo]].copy()
        x.columns = [col.replace('_retorno_esperado', '') for col in x.columns]
        print(x)
        print('===========================================================')

        return pct_change ,retornos, probabilidades, esperancas#.reset_index()


if __name__ == "__main__":
    params = {

        ######## PARÂMETROS DO BACKTEST DE ÚNICO SPLIT E DE SIMULAÇÃO INTERATIVA ############

    'sigla_ativo':              #[   "PETR4.SA",  # Petrobras PN
    #                                 "VALE3.SA",  # Vale ON
    #                                 "ITUB4.SA",  # Itaú Unibanco PN
    #                                 "BBDC4.SA",  # Bradesco PN
    #                                 "BBAS3.SA",  # Banco do Brasil ON
    #                                 "ABEV3.SA",  # Ambev ON
    #                                 "WEGE3.SA",  # WEG ON
    #                                 "PETR3.SA",  # Petrobras ON
    #                                 "ITSA4.SA",  # Itaúsa PN
    #                                 "B3SA3.SA",  # B3 ON
    #                                 "HASH11.SA", # Hashimatsu PN
    #                                 "IVVB11.SA", # IVV Brasil Units
                                    # "CPLE6.SA",  # Copel PN
                                    # "EGIE3.SA",  # Engie Brasil ON
                                    # "TAEE11.SA", # Taesa Units
                                    # "TRPL4.SA",  # Transmissão Paulista PN
                                    # "CSMG3.SA",  # Copasa ON
                                    # "ELET6.SA",  # Eletrobras PN
                                    # "SANB11.SA", # Santander Brasil Units
                                    # "BBSE3.SA",  # BB Seguridade ON
                                    # "ENBR3.SA",  # Energias do Brasil ON
                                    # "SAPR11.SA", # Sanepar Units
                                    # "FLRY3.SA",  # Fleury ON
                                    # "MRFG3.SA",  # Marfrig ON
                                    # "BRAP4.SA",  # Bradespar PN
                                    # "VIVT3.SA",  # Telefônica Brasil ON
                                    # "KLBN11.SA", # Klabin Units
                                    # "GRND3.SA",  # Grendene ON
                                    # "LEVE3.SA",  # Metal Leve ON
                                    # "PSSA3.SA",  # Porto Seguro ON
                                    # "ALUP11.SA", # Alupar Units
                                    # "HYPE3.SA",  # Hypera Pharma ON
                                    # "RANI3.SA",  # Irani Papel ON
                                    # "CYRE3.SA",  # Cyrela ON
                                    # "MULT3.SA",  # Multiplan ON
                                    # "BRML3.SA",  # BR Malls ON
                                    # "CAML3.SA",  # Camil ON
                                    # "TOTS3.SA",  # Totvs ON
                                # ], 
                                ['HASH11.SA', 'IVVB11.SA', 'DIVO11.SA'], # 'BTC-USD', #HASH11.SA,
        'data_inicial': '2023-01-01',
        'data_final': '2024-12-19',#'2024-11-01',
        'intervalo': '1d',
        'proporcao_teste': 0.25,
        'preditores': [
                            # 'rsi_7',
                            # 'rsi_30',
                            # 'razao_mm_exp_7_30',
                            # 'max_div_min',

                            # 'mm_exp_7',
                            # 'mm_exp_30',
                            # 'razao_amplitude',
                            # 'ret_dia_anterior',
                            # 'tendencia'
                            'open_0', 'open_1', 'open_2', 'open_3', 'open_4', 'open_5',
                            'high_0', 'high_1', 'high_2', 'high_3', 'high_4', 'high_5',
                            'low_0', 'low_1', 'low_2', 'low_3', 'low_4', 'low_5',
                            # 'close_0', 
                            'close_1', 'close_2', 'close_3', 'close_4', 'close_5'
                        ],
        'modelo': DecisionTreeClassifier(random_state=42), #LinearRegression(),
        'DEBUG': False,
        'so_compra': True,              # se True, simula operação comprada e zerada. Se False, simula operação comprada e vendida (contratos)
        'IsSimulacaoInterativa': True,  # se True, simula um forward backtesting (interativo). Se False, simula um backtesting com único fold.

        # ATENÇÃO: SÓ USAR STOP_LOSS QUANDO so_compra = True. AINDA DEVE SER IMPLEMENTADO UM STOP_LOSS PARA SINAL DE VENDA
        'stop_loss': {'tipo': 'dinamico', 'valor': 0.03}, #{'tipo': 'fixo', 'valor': 0.15}, # {'tipo': 'dinamico', 'valor': 0.03}, {'tipo': 'desativado'}

        ######## PARÂMETROS EXCLUSIVOS PARA O FOWARD BACKTEST (SIMULAÇÃO INTERATIVA) ########

        'janela_treino': 242

    }

    sim = Simulador(params)
    sim.getPrecos()
    # print(sim.df)
    # if params['IsSimulacaoInterativa'] == True:
    #     sim.runSimulacaoInterativa() # Executa o forward backtesting
    # elif params['IsSimulacaoInterativa'] == False:
    #     sim.runSimulacao() # Executa o backtesting com divisão única de treino e teste

    # sim.getResultados()
    # sim.runGraficos()

    # Estudos
    # sim.getRetornoMedioHorario()

    
    sim.getDescritivaMarkov(n_classes=3, periodo_variacao=1, plotar_grafico=True)
    # print(sim.getOportunidade())

    
    
