import re
from tqdm import tqdm
from tqdm.auto import trange
import time
import os

from typing import List, Dict, Union, Tuple, NoReturn

import pandas as pd
from pandas.core.series import Series
import json

import numpy as np
import string

import nltk    
from nltk import tokenize    
nltk.download('punkt')   

from gensim.models import KeyedVectors

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.model_selection import train_test_split

import keras
from keras.models import load_model, model_from_json, Sequential
from keras.layers import Dense,Conv1D,MaxPooling1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Flatten, LSTM, Bidirectional, Dropout, concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import Input, Model
from keras.callbacks import ModelCheckpoint

import tensorflow as tf

import sentiment_utils as su

import constants


def _read_dataset_imprensa() -> Tuple[np.ndarray, np.ndarray] :
    '''
        Carrega o dataset imprensa retornando dois numpy`s, o primeiro sáo os textos e o segundo os labels
    '''
    lista_texto : List[str] = []
    lista_avaliacao: List[str] = []

    for arq in os.listdir(constants.DATASET_IMPRENSA):
        if not '.csv' in arq:
            continue
        df = pd.read_csv(os.path.join(constants.DATASET_IMPRENSA, arq), sep='|')
        df = df[df["texto_artigo"] != constants.CONTEUDO_INDISPONIVEL]
        df_texto = df[['texto_artigo']]
        df_avaliacao = df[['Avaliação']]
        
        lista_texto_aux = df_texto.astype(str).values.tolist()
        lista_avaliacao_aux = df_avaliacao.astype(str).values.tolist()
        
        for (texto,label) in zip(lista_texto_aux, lista_avaliacao_aux):
            lista_texto.append(texto[0])
            lista_avaliacao.append(label[0])

    return (lista_texto, lista_avaliacao)


def _read_dataset_mencoes(debug: bool = False) -> Tuple[np.ndarray, np.ndarray] :
    '''
        Carrega o dataset mencoes retornando dois numpy`s, o primeiro sáo os textos e o segundo os labels
    '''
    lista_texto : List[str] = []
    lista_avaliacao: List[str] = []
    qtd_vazias: int = 0

    for dir_ano in os.listdir(constants.DATASET_MENCOES):
        full_path: str = os.path.join(constants.DATASET_MENCOES, dir_ano)
        for arq in os.listdir(full_path):
            if not '.csv' in arq:
                continue
            df = pd.read_csv(os.path.join(full_path, arq), sep='|')
            # remover linhas vazias
            df = df.dropna(subset=['content'])
            df = df.drop_duplicates(subset=['content'])
            
            df_texto = df[['content']]
            df_avaliacao = df[['sentiment']]
            
            lista_texto_aux = df_texto.astype(str).values.tolist()
            lista_avaliacao_aux = df_avaliacao.astype(str).values.tolist()
            
            for (texto,label) in zip(lista_texto_aux, lista_avaliacao_aux):
                if len(texto[0].strip()) > 0 or texto[0] == 'nan':
                    #print(texto[0])
                    lista_texto.append(texto[0])
                    lista_avaliacao.append(constants.LABEL_DICT_CONV_MENCOES[int(float(label[0]))])                
    return (lista_texto, lista_avaliacao)


def _load_stopwords():
    """
    This function loads a stopword list from the *path* file and returns a 
    set of words. Lines begining by '#' are ignored.
    """

    # Set of stopwords
    stopwords = set([])

    # For each line in the file
    with open(constants.STOP_WORDS_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if not re.search('^#', line) and len(line.strip()) > 0:
                stopwords.add(line.strip().lower())

    # inclusão dos tokens gerados incorretamente pelo word tokenize
    stopwords.add("``")
    stopwords.add("''")
    # Return the set of stopwords
    return stopwords

def _tokenizar(lista_texto: List[str]) -> List[List[str]]:
    lista_tokens_clear: List[List[str]] = []
    stop_words = _load_stopwords()

    for texto in tqdm(lista_texto, 'Tokenizando... '):
        texto = su.pre_processar_lstm(texto)
        lista_tokens = tokenize.word_tokenize(texto, language='portuguese')
        lista_tokens = [token.lower() for token in lista_tokens if token.lower() not in stop_words and token not in string.punctuation]
        lista_tokens_clear.append(lista_tokens)
        
    return lista_tokens_clear


def _read_embedding(file_name, dim_size, debug: bool = False) -> Tuple[Dict[str,int], List[np.ndarray]]:
    print('Carregando Embedding...')
    with open(file_name,'r', encoding="utf8") as f:
        vocab:Dict[str, int] = {} 
        word_vector: List[np.ndarray] = []
        pos = 0
        for line_number, line in enumerate(f):
            line_ = line.strip() 
            line_word_vec = line_.split()
            
            if (len(line_word_vec) == dim_size + 1):
                if not line_word_vec[0] in vocab:
                    vocab[line_word_vec[0]] = pos
                    word_vector.append(np.array(line_word_vec[1:],dtype=float)) 
                    pos+=1
            else:
                if debug:
                    print(f'Linha {line_number} Ignorada', line_word_vec)
            
    if debug:
        print("Total palavras no embedding :", len(vocab))
    np_word_vector: np.ndarray = np.stack(word_vector)
    return vocab, np_word_vector


def _trata_tokens_ausentes(lista_textos_tokenizados: List[List[str]], 
                          vocab: List[str], 
                          debug: bool = False) -> Tuple[List[str], Dict[str,int]] :
    '''
        Trata tokens ausentes retirando-os da lista. Pode-se tentar salvar o token com o paramentro de salvar

        lista_textos_tokenizados:
            Lista de lista de tokens extraídos de texto
        salvar_ausente:
            Tenta salvar o ausente buscando palavras próximas (utilizando proximidade de edição)
         debug:
            Logar informações sobre o processo

        Returns:
            List[str] - Tokens limpos para o modelo, ou seja, todos estão no embedding
            Dict[str,int] - Dicionário com palavras ausentes e suas respectivas quantidades
    '''

    dict_tokens_ausentes: Dict[str,int] = {}
    
    lista_textos_tokenizados_ajustada: List[List[str]] = []
    for lista_tokens in tqdm(lista_textos_tokenizados, 'Tratamento tokens ausentes...'):
        lista_token_ajustada: List[str] = []
        for token in lista_tokens:
            if token in vocab:
                lista_token_ajustada.append(token)
            else:
                if token in constants.SUBSTITUICOES_COMUNS_PALAVRAS_INCORRETAS:
                    lista_token_ajustada.append(constants.SUBSTITUICOES_COMUNS_PALAVRAS_INCORRETAS[token])
                else:
                    if token in dict_tokens_ausentes:
                        dict_tokens_ausentes[token] +=1
                    else:
                        dict_tokens_ausentes[token] = 1
        lista_textos_tokenizados_ajustada.append(lista_token_ajustada)

    if debug:
        sorted_ausentes = sorted(dict_tokens_ausentes, key=dict_tokens_ausentes.get, reverse=True)
        for i, r in enumerate(sorted_ausentes):
            print(r, dict_tokens_ausentes[r])
            if (i > 20):
                break

    return lista_textos_tokenizados_ajustada



def _converte_tokens_to_id(lista_sentencas_tokenizadas: List[List[str]], 
                          vocab: Dict[str,int], max_sentence_length: int) -> np.ndarray:
    conversao = np.zeros((len(lista_sentencas_tokenizadas), max_sentence_length), dtype='int32')
    
    for count_sentenca,sentenca in enumerate(lista_sentencas_tokenizadas):
        for count_token,token in enumerate(sentenca):
            if(count_token < max_sentence_length):
                conversao[count_sentenca,count_token] = vocab[token]
        
    return conversao


def _carrega_textos_para_treinamento(vocab: Dict[str,int]) -> Tuple[np.ndarray, List[int]]:

    lista_texto_imprensa: List[str]
    lista_label_imprensa: List[str] 
    lista_texto_mencoes: List[str]
    lista_label_mencoes: List[str]
    lista_texto_completa: List[str] = []
    lista_label_completa: List[str] = []


    print('Carregando textos para treinamento...')

    (lista_texto_imprensa, lista_label_imprensa) = _read_dataset_imprensa()
    (lista_texto_mencoes, lista_label_mencoes) = _read_dataset_mencoes()

    lista_texto_completa.extend(lista_texto_imprensa)
    lista_texto_completa.extend(lista_texto_mencoes)
    lista_label_completa.extend(lista_label_imprensa)
    lista_label_completa.extend(lista_label_mencoes)


    lista_tokens_texto : List[List[str]] = _tokenizar(lista_texto_completa)
    lista_label_conv: List[int] = np.array([constants.LABEL_NAMES.index(label) for label in lista_label_completa])
    lista_tokens_texto_preparado: List[List[str]] = _trata_tokens_ausentes(lista_tokens_texto, vocab, debug=True)

    np_tokens_final: np.ndarray = _converte_tokens_to_id(lista_tokens_texto_preparado, vocab, constants.MAX_LSTM_SENTENCE_LENGTH )

    return (np_tokens_final, lista_label_conv)


def treina_modelo_lstm():
    vocab, word_vector = _read_embedding(constants.EMBEDDING_WORD2VEC_300, 300) 

    (np_tokens_final, lista_label_conv) = _carrega_textos_para_treinamento(vocab)

    X_train, X_test, y_train, y_test = train_test_split(np_tokens_final, lista_label_conv,
                                                    test_size=0.1,
                                                    random_state=0,
                                                    stratify=lista_label_conv)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                    test_size=0.1,
                                                    random_state=0,
                                                    stratify=y_train)

    print('Iniciando treinamento...')

    inp = Input(shape=(constants.MAX_LSTM_SENTENCE_LENGTH,))
    x = Embedding(len(vocab), len(word_vector[0]), weights=[word_vector], input_length=constants.MAX_LSTM_SENTENCE_LENGTH, trainable=False)(inp)
    x = LSTM(300, return_sequences=True, recurrent_dropout=0.3)(x)
    #x = LSTM(150, return_sequences=True, recurrent_dropout=0.3)(x)
    #x = Bidirectional(LSTM(300, return_sequences=True, recurrent_dropout=0.3))(x)
    #x = Bidirectional(LSTM(200, return_sequences=True, recurrent_dropout=0.3))(x)
    x = Dropout(0.3)(x)
    #x = Bidirectional(LSTM(150, recurrent_dropout=0.3, return_sequences=True))(x)
    #avg_pool = GlobalAveragePooling1D()(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(300, activation='relu') (x)
    #x = concatenate([max_pool])
    x = Dense(3, activation='softmax') (x)

    model = Model(inputs=inp, outputs=x)
    model.compile(loss='sparse_categorical_crossentropy',
                optimizer=tf.keras.optimizers.Adam(lr=0.01, decay=0.001), 
                metrics=['accuracy'])

    filepath = os.path.join(constants.MODEL_LSTM_PATH, "bndes_sentiment.hdf5")
    checkpoint = ModelCheckpoint(filepath, 
                                monitor='val_accuracy', 
                                verbose=1, 
                                save_best_only=True, 
                                mode='max',
                                save_weights_only=False)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=3)
    callbacks_list = [checkpoint, early_stopping]

    history_double_lstm = model.fit(X_train, 
                                    y_train, 
                                    validation_data=(X_val, y_val),  
                                    batch_size=32, 
                                    epochs=10, 
                                    callbacks=callbacks_list)

    print('Testando o Modelo...')

    filepath = os.path.join(constants.MODEL_LSTM_PATH, "bndes_sentiment.hdf5")
    model_load = tf.keras.models.load_model(filepath)
    evaluate = model_load.evaluate(X_test,y_test)
    print('Test set\n  Loss: {:0.4f}\n  Accuracy: {:0.4f}'.format(evaluate[0],evaluate[1]))

def prever_sentimento(file_textos: str):
    textos: List[str] = []
    vocab, _ = _read_embedding(constants.EMBEDDING_WORD2VEC_300, 300)

    filepath = os.path.join(constants.MODEL_LSTM_PATH, "bndes_sentiment.hdf5")
    model_load = tf.keras.models.load_model(filepath)

    with open(file_textos, 'r', encoding='utf-8') as f:
        for line in f:
            textos.append(line.strip().lower())


    lista_tokens : List[List[str]] = _tokenizar(textos)
    lista_tokens = _trata_tokens_ausentes(lista_tokens, vocab)
    np_tokens: np.ndarray = _converte_tokens_to_id(lista_tokens, vocab, constants.MAX_LSTM_SENTENCE_LENGTH )

    predict_logits = model_load.predict(np_tokens)
    predict = np.argmax(predict_logits, axis=1)
    for index, t in enumerate(textos):
        print(f' Sentimento: {constants.LABEL_NAMES[predict[index]]} /n {t}')
        print(' ===============================================')
