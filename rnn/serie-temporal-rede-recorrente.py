#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A base de dados das ações da petr4 foram obtidas do site Yahoo! finanças
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error


# Importação da base de dados
base = pd.read_csv('petr4.csv')

# Pré-processamento
base = base.dropna()  # remove as linhas que possuem valores NaN

# Iremos utilizar apenas a previsão dos dados da coluna 1 (Atribute Open)
base = base.iloc[:, 1].values  # values significa transformar em numpy array


plt.plot(base)  # exibe os dados da série temporal do atributo (Open) de petr4


periodos = 30  # variável indica que queremos fazer a previsão dos próximos 30 dias
previsao_futura = 1  # indica o horizonte das previsões (ou seja) o valor 1 indica que queremos 1x o valor de periodos


# Para o atributo previsor X queremos fazer um slice na base
# Queremos desde a linha 0 da variável base até o resultado da operação explicada a seguir
# Iremos subtrair o tamanho total da base pelo valor do resto da divisão da base por 30
X = base[0:(len(base) - (len(base) % periodos))]


"""
Criação de Tensores para nossa rede neural recorrente

Criação de batches para a execução mais rápida da nossa rede neural

Criaremos 41 lotes (batches) contendo 30 valores em cada.

O valor -1 significa que não sabemos qual valor receberá para este parâmetro da dimensionalidade

Passamos a variável 'periodos' que significa que queremos que nossos batches (lotes)
tenham o tamanho total de 30 registros em cada lote com 1 coluna

Explicando, em cada índice desses 41 temos 30 registros (linhas) com uma coluna apenas

Nosso tensor então terá um total de 3 dimensões
"""
# A seguir mudaremos a dimensionalidade da variável X, onde não mudaremos as linhas
# Teremos 41 registros com a dimensionalidade 30 linhas e 1 coluna
# Se multiplicar 41 x 30 o valor é 1230 que é a qtd de registros de X
X_batches = X.reshape(-1, periodos, 1)  # mudando a dimensionalidade da variável X



"""
O nosso atributo target (alvo) / classe será o valor posterior a 30 valores do atributo previsor
Ou seja, a cada 30 valores (1 mês) o valor alvo será o valor subsequente
A ideia é: para cada 30 dias de valores de abertura da bolsa, queremos prever
o valor subsequente desses 30 dias

Lembre-se que nosso atributo previsor possui 41 indices que são nossos batches
onde cada um possui 30 registros com 1 coluna
se abrir a visualização da variável e ir para o eixo 2 (ultimo), vamos perceber
que para cada linha teremos 30 colunas
Ou seja, cada linha representa 1 batch e cada coluna representa o valor de abertura
da bolsa para 1 dia

Portanto, queremos prever o valor subsequente
Para fazer o treinamento, vamos fazer um slice na base de dados onde o atributo target
receberá os valores subsequentes a cada periodo de 30 dias
"""

# Fazemos um slice na base original iniciando da linha 1 e somando 1 que é a previsao_futura
y = base[1:(len(base) - (len(base) % periodos)) + previsao_futura]

# Fazendo reshape para ficar na mesma dimensionalidade de X_batches
y_batches = y.reshape(-1, periodos, 1)


"""
O último valor de cada batch de y_batches será o valor da previsão de cada batch de X_batches
"""


# Divisão da base de dados de teste

# para a divisão da base de dados de teste, faremos o comando abaixo
# este comando significa que iremos utilizar os índices negativos, ou seja
# começaremos do fim para o início
# Estamos utilizando o sinal de negativo para indicar que o resultado da soma
# será um valor negativo
# O resultado indica que pegaremos do índice -31 até o final
# Ou seja, como pegamos para treino do ínicio até os ultimos 30
# Então devemos pegar aqueles dados que não foram treinados
X_teste = base[-(periodos + previsao_futura):]


# Pegando apenas do primeiro até o 30 valor
X_teste = X_teste[:periodos]

# Reshape X_teste
X_teste = X_teste.reshape(-1, periodos, 1)

y_teste = base[-(periodos):]
y_teste = y_teste.reshape(-1, periodos, 1)


# Como o TensorFlow trabalha com a estrutura de grafos, executaremos o comando
# abaixo para resetar grafos em memória
tf.reset_default_graph()

entradas = 1  # o valor 1 significa que estamos trabalhando com apenas 1 atributo previsor que é o "Open"
neuronios_oculta = 100  # quantidade de neurônios na camada oculta (valor a ser testado)
neuronios_saida = 1  # como estamos trabalhando com um problema de regressão, estamos prevendo a saída de 1 valor

# criação dos placeholders

# A estrutura de dados do nosso placeholder será desconhecida para a primeira dimensão (None)
# O shape do nosso placeholder será [batch_size, 30, 1]
xph = tf.placeholder(tf.float32, [None, periodos, entradas])
yph = tf.placeholder(tf.float32, [None, periodos, neuronios_saida])  # se fosse um problema com mais de uma possível saída, o valor seria 2 para "neuronios_saida"


# Criação de célula

# Célula é a estrutura que faz a passagem de valor para si mesmo
# Conceito utilizado em redes neurais recorrentes
# Definindo que cada célula terá 100 neuronios na camada oculta
# Indica que cada célula fará um loop e enviara os dados para si mesmo 100x

# Fpo comentado abaixo que é utilizado uma Célula de Basic RNN Simples
#celula = tf.contrib.rnn.BasicRNNCell(num_units = neuronios_oculta, activation = tf.nn.relu)

# Utilizando RNN LSTM
# celula = tf.contrib.rnn.LSTMCell(num_units = neuronios_oculta, activation = tf.nn.relu)

# nosso objeto celula irá receber a própria celula com 100 neuronios na camada de saída e vão se ligar a um neurônio na camada de saída
# isso deve ser feito para não gerar erros indicando que teremos apenas 1 neurônio na camada de saída
# celula = tf.contrib.rnn.OutputProjectionWrapper(celula, output_size = 1)



# criação de funções para a criação de células
def cria_uma_celula():
    return tf.contrib.rnn.LSTMCell(num_units = neuronios_oculta, activation = tf.nn.relu)

def cria_varias_celulas():
    celulas = tf.nn.rnn_cell.MultiRNNCell([cria_uma_celula() for i in range(4)])
    return tf.contrib.rnn.DropoutWrapper(celulas, output_keep_prob = 0.1)


# Agora, estamos criando o nosso objeto celula contendo várias celulas que foram retornadas da função cria_varias_celulas()
celula = cria_varias_celulas()

celula = tf.contrib.rnn.OutputProjectionWrapper(celula, output_size = 1)

# recebimento da saida da RNN
# veja que a função retorna 2 valores, onde o segundo será descartado, como indica o _
# Veja que inserimos apenas 1 celula que é a quantidade de camadas ocultas que será 1 camada oculta
saida_rnn, _ = tf.nn.dynamic_rnn(celula, xph, dtype = tf.float32)

# Criação da função do erro
# passamos como parâmetro as saidas reais que é o "labels" indicado pelo nosso placeholder yph que receberá os dados da saída real
erro = tf.losses.mean_squared_error(labels = yph, predictions = saida_rnn)

# Criação do otimizador
otimizador = tf.train.AdamOptimizer(learning_rate=0.001)

# criação da variável treinamento que recebe o resultado da otimização e minimizar o resultado do erro
treinamento = otimizador.minimize(erro)


"""
Criação de uma Sessão para a execução das operações definidas pelo TensorFlow
"""

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # inicializando as variáveis do TensorFlow
    
    # TREINAMENTO
    # definindo o treinamento em 1000 épocas
    for epoca in range(1000):
        _, custo = sess.run([treinamento, erro], feed_dict = {xph: X_batches, yph: y_batches})
        
        if epoca % 100 == 0:
            print(epoca + 1, 'erro', custo)  # a cada 100 épocas é exibido o erro atual durante o treinamento

    
    # PREVISOES
    # passamos para o placeholder os dados de teste
    previsoes = sess.run(saida_rnn, feed_dict = {xph: X_teste})


# Realizando o cálculo do erro que é a subtração do valor real com valor previsto
# Realizando transformações
y_teste.shape  # y_teste contém os valores reais
y_teste2 = np.ravel(y_teste)  # diminui a dimensionalidade

previsoes.shape  # previsoes contém os valores previstos
previsoes2 = np.ravel(previsoes)


# O resultado do cálculo erro absoluto é a subtração do valor real com o valor previsto
mae = mean_absolute_error(y_teste2, previsoes2)    


# Visualização - utilizando símbolos
plt.plot(y_teste2, '*', markersize = 10, label = 'Valor Real')  # visualizando os valores reais da ação dos últimos 30 valores
plt.plot(previsoes2, 'o', label = 'Previsões')  # visualizando os valores previstos para os 30 últimos valores
plt.legend()

# Visualização - utilizando reta
plt.plot(y_teste2, label = 'Valor Real')  # visualizando os valores reais da ação dos últimos 30 valores
plt.plot(previsoes2, label = 'Previsões')  # visualizando os valores previstos para os 30 últimos valores
plt.legend()


"""
RESULTADOS UTILIZANDO APENAS 1 CAMADA OCULTA COM 100 NEURÔNIOS:
    
Para esta base de dados, o Resultado para uma RNN Simples retornou um resultado melhor
do que utilizando uma LSTM

Resultado MAE (Mean Absolute Error):
    - RNN Simples: 0.16
    - LSTM:        0.21
    
    
Para alterar o tipo de RNN, comente uma das linhas que está na BasicRNNCell
para LSTM


RESULTADOS UTILIZANDO 4 CAMADAS OCULTAS COM LSTM

O resultado ficou pior.

Resultado MAE (Mean Absolute Error):
    - LSTM:     0.27
    
    

RESULTADO UTILIZANDO DROPOUT COM 4 CAMADAS OCULTAS COM LSTM

O resultado ficou pior.

Resultado MAE (Mean Absolute Error):
    - LSTM:     3.60
    
"""
