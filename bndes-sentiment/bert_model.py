
import re
from tqdm import tqdm
from tqdm.auto import trange
import time
import os

from typing import List, Dict, Union, Tuple, NoReturn

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

import keras

import transformers
from transformers import AutoTokenizer  # Or BertTokenizer
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification
from datasets import load_metric

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F

import gc

import sentiment_utils as su
import constants



RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BNDESSentimentDataset(torch.utils.data.Dataset):

  def __init__(self, textos, sentimentos, tokenizer, max_len):
    self.lista_texto = textos
    self.lista_sentimento = sentimentos
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.lista_texto)
  
  def __getitem__(self, item):
    texto = str(self.lista_texto[item])
    sentimento = self.lista_sentimento[item]

    encoding = self.tokenizer.encode_plus(
      texto,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'texto': texto,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'labels': torch.tensor(sentimento, dtype=torch.long)
    }

# Liberar e monitorar memória da GPU
def _destroy_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()

def _destroy_tokenizer(tokenizer):
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def _get_gpu_memory_status():
    total = (torch.cuda.get_device_properties(0).total_memory)/(1024 **2)
    reserved = (torch.cuda.memory_reserved(0))/(1024 **2)
    allocated = (torch.cuda.memory_allocated(0))/(1024 **2)
    return f"Total: {total:.2f} | Reserved: {reserved:.2f} | Allocated: {allocated:.2f}"


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

def _carrega_texto_treinamento() -> Tuple[List[str], List[str]]:
    '''
        Carrega todos os textos necessários para treinamento

        Returns:
            List[str] - Textos
            List[str] - Labels
    '''

    lista_texto_imprensa: List[str]
    lista_label_imprensa: List[str] 
    lista_texto_mencoes: List[str]
    lista_label_mencoes: List[str]
    lista_texto_completa: List[str] = []
    lista_texto_completa_final: List[str] = []
    lista_label_completa: List[str] = []


    (lista_texto_imprensa, lista_label_imprensa) = _read_dataset_imprensa()
    (lista_texto_mencoes, lista_label_mencoes) = _read_dataset_mencoes()

    lista_texto_completa.extend(lista_texto_imprensa)
    lista_texto_completa.extend(lista_texto_mencoes)
    lista_label_completa.extend(lista_label_imprensa)
    lista_label_completa.extend(lista_label_mencoes)


    for texto in tqdm(lista_texto_completa,'Limpando textos para entrada no modelo....'):
        lista_texto_completa_final.append(su.pre_processar_bert(texto))

    return (lista_texto_completa_final, lista_label_completa)


def _compute_metrics(eval_pred):
   load_accuracy = load_metric("accuracy")
   load_f1 = load_metric("f1")
  
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1 = load_f1.compute(predictions=predictions, references=labels, average='macro')["f1"]
   return {"accuracy": accuracy, "f1": f1}


def treina_modelo_bert():
    print('Carregando textos...')
    (lista_texto_completa,lista_label_completa) = _carrega_texto_treinamento()
    lista_label_conv: List[int] = np.array([constants.LABEL_NAMES.index(label) for label in lista_label_completa])
    print(f'Texto {len(lista_texto_completa)} Label {len(lista_label_conv)}')

    model = AutoModelForSequenceClassification.from_pretrained(constants.BASE_BERT_MODEL, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(constants.BASE_BERT_MODEL, do_lower_case=False)

    X_train, X_test, y_train, y_test = train_test_split(lista_texto_completa, lista_label_conv,
                                                    test_size=0.1,
                                                    random_state=0,
                                                    stratify=lista_label_conv)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                    test_size=0.1,
                                                    random_state=0,
                                                    stratify=y_train)


    train_dataset = BNDESSentimentDataset(textos=X_train,sentimentos=y_train,tokenizer=tokenizer,max_len=constants.BERT_MAX_LEN)
    val_dataset = BNDESSentimentDataset(textos=X_val,sentimentos=y_val,tokenizer=tokenizer,max_len=constants.BERT_MAX_LEN)
    test_dataset = BNDESSentimentDataset(textos=X_test,sentimentos=y_test,tokenizer=tokenizer,max_len=constants.BERT_MAX_LEN)

    print('Iniciando treinamento....')

    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # total number of training epochs
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir= constants.MODEL_TRAINED_LOG,            # directory for storing logs
        logging_steps=500,
        evaluation_strategy='steps',
        save_strategy='steps',
        load_best_model_at_end=True)

    trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset ,            # evaluation dataset
            compute_metrics=_compute_metrics)

    trainer.train()

    trainer.save_model(constants.MODEL_BERT_PATH)
    _destroy_model(model)
    _destroy_tokenizer(tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(constants.MODEL_BERT_PATH, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(constants.BASE_BERT_MODEL, do_lower_case=False)
    model.to('cuda')

    loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    loop = tqdm(loader, leave=True)
    final_output_loss = []
    final_output_logits = []

    for it, batch in enumerate(loop):

        input_ids = batch['input_ids'].to('cuda')
        attention_mask = batch['attention_mask'].to('cuda')
        labels = batch['labels'].to('cuda')

        outputs = model(input_ids=input_ids, 
                        attention_mask=attention_mask, 
                        labels=labels)
        
        final_output_loss.append(outputs.loss.detach().to('cpu').numpy())
        final_output_logits.append(outputs.logits.detach().to('cpu').numpy())

        input_ids.detach()
        attention_mask.detach()
        labels.detach
        labels = None
        input_ids = None
        attention_mask = None
        outputs.logits = None
        outputs.loss = None
        outputs = None
        gc.collect()
        torch.cuda.empty_cache()

    _destroy_model(model)
    _get_gpu_memory_status()

    test_preds = np.vstack(final_output_logits)
    test_preds = np.argmax(test_preds, axis=-1)

    y_true = np.array(y_test).ravel()
    y_pred = test_preds.ravel()

    score = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    print('F1 Score (Macro) - ', score)
    score = accuracy_score(y_true=y_true, y_pred=y_pred)
    print('Acurácia', score)



def prever_sentimento(file_textos: str):
    textos: List[str] = []
    model = AutoModelForSequenceClassification.from_pretrained(constants.MODEL_BERT_PATH, num_labels=3)
    tokenizer = AutoTokenizer.from_pretrained(constants.BASE_BERT_MODEL, do_lower_case=False)
    model.to('cuda')

    with open(file_textos, 'r', encoding='utf-8') as f:
        for line in f:
            textos.append(su.pre_processar_bert(line))


    encoding = tokenizer.batch_encode_plus(
      textos,
      add_special_tokens=True,
      max_length=constants.BERT_MAX_LEN,
      return_token_type_ids=False,
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt'
    )

    model_predict = model(encoding['input_ids'].to('cuda'), encoding['attention_mask'].to('cuda'))
    model_predict_converted = F.softmax(model_predict.logits, dim=1).cpu().detach().numpy()
    predict = np.argmax(model_predict_converted, axis=1)
    for index, t in enumerate(textos):
        print(f' Sentimento: {constants.LABEL_NAMES[predict[index]]} \n {t}')
        print(' ===============================================')
    