'''
    Instruções
    1) Em nosso projeto utilizamos o compartilhamento do Google Drive que continha os textos clasificados  e após o treinamento salvamos o modelo também no Google Drive.
    2) A execução em linha de comando pode ser realizada da seguinte forma:
        - Treinamento BERT -> python bndes-sentiment.py bert treino
        - Treinamento LSTM -> python bndes-sentiment.py lstm treino
        - Predição com o modelo BERT (textos em um arquivo) -> python bndes-sentiment.py bert predicao <nome-arquivo.txt>
        - Predição com o modelo LSTM (textos em um arquivo) -> python bndes-sentiment.py lstm predicao <nome-arquivo.txt>
    
'''


import sys
import numpy

import lstm_model
import bert_model

TIPO_REDE_LSTM = 'lstm'
TIPO_REDE_BERT = 'bert'

OBJETIVO_TREINO = 'treino'
OBJETVIVO_PREDICAO = 'predicao'



def valida_parametros(tipo_rede: str, objetivo: str) -> None:
'''
    Valida os parametros recebidos pela linha de comando
'''
    if (tipo_rede.lower() not in [TIPO_REDE_LSTM,TIPO_REDE_BERT]):
        mensagem = f'Tipo de rede inválida {tipo_rede}'
        raise Exception(mensagem)
    if (objetivo.lower() not in [OBJETIVO_TREINO, OBJETVIVO_PREDICAO]):
        mensagem = f'Objetivo inválido {objetivo}'
        raise Exception(mensagem)


"""
    Param1: Tipo de Rede
        - lstm: LSTM
        - bert: BERT
    Param2: Objetivo
        - train: Treinar o modelo
        - evaluate: realizar inferencia sobre o texto informado 
"""
def main():
    tipo_rede: str = sys.argv[1]
    objetivo: str = sys.argv[2]
    if len(sys.argv) > 3 and objetivo == OBJETVIVO_PREDICAO :
        file_textos: str = sys.argv[3]
    
    valida_parametros(tipo_rede=tipo_rede, objetivo=objetivo)


    if tipo_rede.lower() == TIPO_REDE_LSTM:
        if objetivo.lower() == OBJETIVO_TREINO: 
            lstm_model.treina_modelo_lstm()
        else:
            lstm_model.prever_sentimento(file_textos)
    elif tipo_rede.lower() == 'bert':
        if objetivo.lower() == OBJETIVO_TREINO: 
            bert_model.treina_modelo_bert()
        else:
            bert_model.prever_sentimento(file_textos)
    

if __name__ == '__main__':
   main()