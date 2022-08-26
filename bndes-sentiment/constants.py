from typing import List, Dict, Union, Tuple, NoReturn
import os

GDRIVE_PATH:str = '/content/drive/MyDrive'
DATASET_ROOT_PATH: str = os.path.join(GDRIVE_PATH, 'dataset', 'projeto-final')
DATASET_IMPRENSA: str = os.path.join(DATASET_ROOT_PATH, 'imprensa')
DATASET_MENCOES: str = os.path.join(DATASET_ROOT_PATH, 'mencoes')
CONTEUDO_INDISPONIVEL: str = 'CONTEUDO_INDISPONIVEL'
STOP_WORDS_FILE: str = os.path.join(DATASET_ROOT_PATH, 'stop-words.txt')

EMBEDDING_GLOVE_50 : str = os.path.join(GDRIVE_PATH, 'model', 'embedding','glove_s50.txt')
EMBEDDING_WORD2VEC_300: str = os.path.join(GDRIVE_PATH, 'model', 'embedding','cbow_s300.txt')
EMBEDDING_WORD2VEC_600: str = os.path.join(GDRIVE_PATH, 'model', 'embedding','cbow_s600.txt')
EMBEDDING_WORD2VEC_1000: str = os.path.join(GDRIVE_PATH, 'model', 'embedding','cbow_s1000.txt')
EMBEDDING_GLOVE_1000: str = os.path.join(GDRIVE_PATH, 'model', 'embedding','glove_s1000.txt')

LABEL_NAMES: List[str] = ['Negativa', 'Neutra', 'Positiva']
LABEL_DICT_CONV_MENCOES : Dict[int,str] = {-5:'Negativa', 0:'Neutra', 5:'Positiva'}

MAX_LSTM_SENTENCE_LENGTH: int = 1000 
MODEL_LSTM_PATH: str = os.path.join(GDRIVE_PATH, 'model', 'lstm', 'projeto-final')


MODEL_BERT_PATH: str = os.path.join(GDRIVE_PATH, 'model', 'projeto-final', 'bert')
MODEL_TRAINED_LOG: str = os.path.join(MODEL_BERT_PATH, 'trainer.log')

BERT_BATCH_SIZE = 16
BERT_MAX_LEN = 512

BASE_BERT_MODEL: str = 'neuralmind/bert-base-portuguese-cased'

# palavras comumente incorretas
SUBSTITUICOES_COMUNS_PALAVRAS_INCORRETAS: Dict[str,str] = {
    'covid-00': 'coronavírus',  'privatizacao':  'privatização', 'leilao':  'leilão',
    'inflacao':  'inflação', 'bilhao':  'bilhão', 'concessoes': 'concessões',
    'aprovacao': 'aprovação', 'covid': 'coronavírus', 'desestatizacao': 'desestatização',
    'governanca': 'governança', 'atuacao': 'atuação', 'emissoes': 'emissões', 
    'manutencao': 'manutenção', 'licitacao': 'licitação', 'protecao': 'proteção',
    'emissao': 'emissão', 'contratacao': 'contratação', 'aquisicao': 'aquisição',
    'arrecadacao': 'arrecadação', 'votacao': 'votação', 'ampliacao': 'ampliação',
    'negociacao': 'negociação', 'vacinacao': 'vacinação', 'inadimplencia': 'inadimplência', 
    'poupanca': 'poupança', 'realizacao': 'realização', 'suspensao': 'suspensão', 
    'preservacao': 'preservação', 'estruturacao': 'estruturação', 'fiscalizacao': 'fiscalização',
    'capitalizacao': 'capitalização', 'conservacao': 'conservação', 'prestacao': 'prestação',
    'cobranca': 'cobrança', 'transicao': 'transição', 'remuneracao': 'remuneração',
    'liberacao': 'liberação', 'discussoes': 'discussões', 'universalizacao': 'universalização',
    'aculpa': 'culpa', 'deverao': 'deverão', 'elaboracao': 'elaboração',
    'contribuicao': 'contribuição', 'mineracao': 'mineração', 'adesao': 'adesão',
    'modernizacao': 'modernização','regulacao': 'regulação', 'projecao': 'projeção',
    'regulatorio': 'regulatório', 'centrao': 'centrão', 'avancos': 'avanços',
    'climatica': 'climática', 'variacao': 'variação', 'implantacao': 'implantação',
    'implementacao': 'implementação', 'projecoes': 'projeções', 'senadorhumberto': 'senador',
    'trilhao': 'trilhão', 'reeleicao':  'reeleição', 'restricoes': 'restrições',
    'elevacao': 'elevação', 'percepcao': 'percepção', 'importacao': 'importação',
    'exportacao':  'exportação', 'valorizacao':  'valorização', 'licitacoes': 'licitações', 
    'adocao': 'adoção', 'trilhoes': 'trilhões', 'sustentaveis': 'sustentáveis',
    'aviacao': 'aviação', 'pregao': 'pregão', 'recessao': 'recessão', 'reacao': 'reação',
    'mineconomia' : 'ministro', 'regulamentacao' : 'regulamentação', 'movimentacao' : 'movimentação',
    'p/': 'para'
}
