
import re
import spacy
import string

pln = spacy.load('pt_core_news_sm')
stop_words = spacy.lang.pt.stop_words.STOP_WORDS

def get_top_category(predictions, categories):
    """
    Retorna a categoria com a maior probabilidade.

    Args:
    - predictions: Lista ou vetor numpy com as probabilidades preditas pelo modelo.
    - categories: Lista com as categorias correspondentes às probabilidades.

    Returns:
    - A categoria com a maior probabilidade.
    """
    # Encontra o índice da maior probabilidade
    max_index = predictions.argmax()
    # Retorna a categoria correspondente ao índice
    return categories[max_index]
  
def get_top_categories(predictions, categories):
    """
    Retorna as categorias com as maiores probabilidades para cada sentença.

    Args:
    - predictions: Matriz numpy (ou lista de listas) onde cada linha contém as probabilidades preditas pelo modelo.
    - categories: Lista com as categorias correspondentes às probabilidades.

    Returns:
    - Uma lista de categorias, onde cada elemento corresponde à maior probabilidade de uma sentença.
    """
    top_categories = [categories[pred.argmax()] for pred in predictions]
    return top_categories



def preprocessamento(texto):
  # Letras minúsculas
  texto = texto.lower()

  # Nome do usuário
  texto = re.sub(r"@[A-Za-z0-9$-_@.&+]+", ' ', texto)

  # URLs
  texto = re.sub(r"https?://[A-Za-z0-9./]+", ' ', texto)
  
  #Removendo todos os :d (sorriso) ou sorriso:p
  texto = texto.replace(':d','')
  texto = texto.replace(':p','')

  #Removendo os digitos
  texto = re.sub(r'\d+','',texto)

  #Removendo caracteres repetidos 3 ou mais vezes
  texto = re.sub(r'(\w)\1(\1+)',r'\1',texto)
  
  # Espaços em branco
  texto = re.sub(r" +", ' ', texto)
  
  #Removendo os espaços do inicio e final de cada frase
  texto = texto.strip()

  # Emoticons
  lista_emocoes = {':)': 'emocaopositiva',
                 ':d': 'emocaopositiva',
                 ':(': 'emocaonegativa',
                 ':|': 'emoçaoneutra',
                 ':/': 'emoçaoneutra'}

  for emocao in lista_emocoes:
    texto = texto.replace(emocao, lista_emocoes[emocao])

  # Lematização
  documento = pln(texto)

  lista = []
  for token in documento:
    lista.append(token.lemma_)

  # Stop words e pontuações
  lista = [palavra for palavra in lista if palavra not in stop_words and palavra not in string.punctuation]
  lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])

  return lista