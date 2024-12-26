# Projeto de Análise de Sentimentos

Este projeto foi desenvolvido para realizar a classificação de sentimentos com base em textos utilizando técnicas de Processamento de Linguagem Natural (PLN) e Redes Neurais. O pipeline do projeto abrange o pré-processamento de dados, treinamento do modelo, e predições com base nos dados processados.

---

## **Estrutura do Projeto**

1. **1- Preprocessamento.ipynb**
   - Contém as etapas de pré-processamento dos textos.
   - Limpeza dos dados, remoção de caracteres especiais, URLs, nomes de usuários, e aplicação de lematização.

2. **2- sentimento_train.ipynb**
   - Inclui o treinamento do modelo.
   - Utiliza embeddings, camadas de atenção, LSTMs e GRUs para classificação de sentimentos.
   - Realiza a divisão entre conjunto de treinamento e validação.

3. **3- Test_validation.ipynb**
   - Avalia o modelo treinado em dados de teste.
   - Calcula métricas de desempenho como acurácia, precisão, revocação e F1-score.

---

## **Dependências**
Certifique-se de ter as seguintes dependências instaladas no seu ambiente:

- Python 3.7+
- TensorFlow 2.10+
- SpaCy
- scikit-learn
- NumPy
- pandas
- matplotlib
- nltk
- re

### Instalação das Dependências:
```bash
pip install tensorflow spacy scikit-learn numpy pandas matplotlib nltk
```

---

## **Pipeline do Projeto**

### 1. Pré-Processamento

O arquivo **1- Preprocessamento.ipynb** contém as seguintes etapas:

- **Letras Minúsculas**: Converte todo o texto para minúsculas.
- **Remoção de URLs**: Remove links da internet.
- **Remoção de Nomes de Usuários**: Substitui menções como `@usuario`.
- **Lematização**: Normaliza as palavras para suas formas canônicas.
- **Stopwords**: Remove palavras irrelevantes.
- **Padding e Tokenização**: Prepara os dados para entrada no modelo.

### 2. Treinamento do Modelo

No arquivo **2- sentimento_train.ipynb**, é utilizado um modelo baseado em Redes Neurais, que inclui:

- **Camada de Embedding**: Inicializa embeddings utilizando FastText ou embedding aleatório.
- **Camadas Bidirecionais**: LSTMs e GRUs para capturar relações contextuais.
- **Atenção**: Melhora o foco do modelo em partes relevantes da entrada.
- **Dropout**: Reduz overfitting.
- **Treinamento**: Otimiza o modelo com `categorical_crossentropy` e Adam.

### 3. Avaliação e Testes

No arquivo **3- Test_validation.ipynb**, é feita a avaliação do modelo em dados de teste, com:

- **Métricas de Avaliação**:
  - Acurácia
  - Precisão
  - Revocação
  - F1-Score

---

## **Uso do Modelo**

### Classe de Predição
A classe **`Prediction`** permite carregar o modelo treinado, aplicar pré-processamento e realizar predições em novas entradas.

#### Exemplo de Uso:

```python
from prediction import Prediction

# Inicialização
predictor = Prediction(path_model='modelos/26_12_2024')

# Predição para uma única frase
sentence = "Eu adorei o produto!"
prediction = predictor.run_predict(sentence)
print("Predição:", prediction)

# Predição para múltiplas frases
sentences = [
    "O serviço foi horrível.",
    "Gostei bastante do atendimento!",
    "Foi razoável, nada especial."
]
predictions = predictor.run_predict_all(sentences)
for sentence, category in zip(sentences, predictor.get_top_categories(predictions)):
    print(f"'{sentence}' -> Categoria: {category}")
```

---

## **Estrutura de Arquivos do Modelo**

1. **modelo_atencao.h5**: Arquivo do modelo treinado.
2. **tokenizer.pickle**: Tokenizador utilizado para converter textos em sequências numéricas.
3. **categorias.pickle**: Mapeamento de índices para categorias (ex.: `['negativo', 'neutro', 'positivo']`).

---

## **Contribuições**
- Para sugerir melhorias, entre em contato ou abra uma solicitação no repositório do projeto.

## **Licença**
Este projeto está licenciado sob a [MIT License](https://opensource.org/licenses/MIT).

