## Instruções e Comentários sobre o Projeto

* Os datasets utilizados no treinamento não podem ser publicados porque embora os textos sejam públicos, a sentimentalização obtida é protegida por sigilo.  
* Os notebooks utilizados no projeto estão disponiveis na pasta [notebooks](/notebooks). O projeto foi construído inicialmente em notebooks do Colab e posteriormente trasnformado em um programa para ser acionado pela linha de comando.  
* Foi utilizado compartilhamento do Google Drive para os textos clasificados, onde também foram salvos o modelo LSTM e BERT.  
* A execução em linha de comando pode ser realizada da seguinte forma:  
  * Treinamento BERT -> python bndes-sentiment.py bert treino  
  * Treinamento LSTM -> python bndes-sentiment.py lstm treino  
  * Predição com o modelo BERT (textos em um arquivo) -> python bndes-sentiment.py bert predicao <nome-arquivo.txt>  
  * Predição com o modelo LSTM (textos em um arquivo) -> python bndes-sentiment.py lstm predicao <nome-arquivo.txt>  