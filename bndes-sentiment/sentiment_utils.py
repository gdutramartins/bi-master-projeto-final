import re
from typing import List, Dict, Union, Tuple, NoReturn
import constants


def _remove_links(tweet):
    """Takes a string and removes web links from it"""
    tweet = re.sub(r'http\S+', '', tweet)   # remove http links
    tweet = re.sub(r'bit.ly/\S+', '', tweet)  # remove bitly links
    tweet = tweet.strip('[link]')   # remove [links]
    tweet = re.sub(r'pic.twitter\S+','', tweet)
    return tweet

def _remove_users(tweet):
    """Takes a string and removes retweet and @user information"""
    tweet = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+[:]*)', '', tweet)  # remove re-tweet
    tweet = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+[:]*)', '', tweet)  # remove tweeted at
    return tweet

def _remove_hashtags(tweet):
    """Takes a string and removes any hash tags"""
    tweet = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', tweet)  # remove hash tags
    return tweet

def _remove_av(tweet):
    """Takes a string and removes AUDIO/VIDEO tags or labels"""
    tweet = re.sub('VIDEO:', '', tweet)  # remove 'VIDEO:' from start of tweet
    tweet = re.sub('AUDIO:', '', tweet)  # remove 'AUDIO:' from start of tweet
    return tweet

def _trata_bndes(tweet):
    """Trata o nome do BNDES, que aparece muito e não esá"""
    tweet = re.sub('BNDES', 'banco', tweet)  
    tweet = re.sub('BNDS', 'banco', tweet)  
    
    return tweet

def _trata_erros_escrita_comuns(tweet):
    for key in constants.SUBSTITUICOES_COMUNS_PALAVRAS_INCORRETAS:
        tweet = re.sub(key, constants.SUBSTITUICOES_COMUNS_PALAVRAS_INCORRETAS[key], tweet)
    return tweet

def pre_processar_bert(text: str, trata_nome_bndes: bool = False) -> str:
    
    #text = _remove_links(text)
    #text = _remove_users(text)
    #text = _remove_hashtags(text)
    #text = _remove_av(text)
    #if trata_nome_bndes:
    #    text = _trata_bndes(text)
    text = _trata_erros_escrita_comuns(text)

    return text.strip()

def pre_processar_lstm(text: str, trata_nome_bndes: bool = False) -> str:
    re_remove_brackets = re.compile(r'\{.*\}')
    re_remove_html = re.compile(r'<(\/|\\)?.+?>', re.UNICODE)
    re_transform_numbers = re.compile(r'\d', re.UNICODE)
    re_transform_emails = re.compile(r'[^\s]+@[^\s]+', re.UNICODE)
    re_transform_url = re.compile(r'(http|https)://[^\s]+', re.UNICODE)
    # Different quotes are used.
    re_quotes_1 = re.compile(r"(?u)(^|\W)[‘’′`']", re.UNICODE)
    re_quotes_2 = re.compile(r"(?u)[‘’`′'](\W|$)", re.UNICODE)
    re_quotes_3 = re.compile(r'(?u)[‘’`′“”]', re.UNICODE)
    re_dots = re.compile(r'(?<!\.)\.\.(?!\.)', re.UNICODE)
    re_punctuation = re.compile(r'([,";:]){2},', re.UNICODE)
    re_hiphen = re.compile(r' -(?=[^\W\d_])', re.UNICODE)
    re_tree_dots = re.compile(u'…', re.UNICODE)
    # Differents punctuation patterns are used.
   # re_punkts = re.compile(r'(\w+)([%s])([ %s])' %
    #                    (punctuations, punctuations), re.UNICODE)
    #re_punkts_b = re.compile(r'([ %s])([%s])(\w+)' %
    #                        (punctuations, punctuations), re.UNICODE)
    #re_punkts_c = re.compile(r'(\w+)([%s])$' % (punctuations), re.UNICODE)
    re_changehyphen = re.compile(u'–')
    re_doublequotes_1 = re.compile(r'(\"\")')
    re_doublequotes_2 = re.compile(r'(\'\')')
    re_trim = re.compile(r' +', re.UNICODE)
    
    """Apply all regex above to a given string."""
    text = text.lower()
    text = text.replace('\xa0', ' ')
    text = re_tree_dots.sub('...', text)
    text = re.sub('\.\.\.', '', text)
    text = re_remove_brackets.sub('', text)
    text = re_changehyphen.sub('-', text)
    text = re_remove_html.sub(' ', text)
    text = re_transform_numbers.sub('0', text)
    text = re_transform_url.sub('URL', text)
    text = re_transform_emails.sub('EMAIL', text)
    text = re_quotes_1.sub(r'\1"', text)
    text = re_quotes_2.sub(r'"\1', text)
    text = re_quotes_3.sub('"', text)
    text = re.sub('"', '', text)
    text = re_dots.sub('.', text)
    text = re_punctuation.sub(r'\1', text)
    text = re_hiphen.sub(' - ', text)
    #text = re_punkts.sub(r'\1 \2 \3', text)
    #text = re_punkts_b.sub(r'\1 \2 \3', text)
    #text = re_punkts_c.sub(r'\1 \2', text)
    text = re_doublequotes_1.sub('\"', text)
    text = re_doublequotes_2.sub('\'', text)
    text = re_trim.sub(' ', text)
    
    return text.strip()
