import nltk
import numpy as np

from nltk.stem.porter import PorterStemmer # Есть различные Стеммеры, мы воспользуемся этим

#nltk.download('punkt') # Используется при первом запуске для работы word_tokenize

# tokenize
def tokenize(sentence):
    """
    Разделение предложения на отдельные слова/токены
    Токенами могут быть: слова, пунктуация, цифры
    """
    return nltk.word_tokenize(sentence)

# stem
stemmer = PorterStemmer()
def stem(word):
    """
    stemming = ищет корневую форму слова
    Пример:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower()) # К слову будет применен Стеммер и слово будет преобразовано в нижний регистр при необходимости

# bag of words
def bag_of_words(tokenized_sentence, all_words):
    """
    Вернет массив "bag_of_words":
        1 - ставится если слово есть в предложении;
        0 - ставится если слова нет в предложении.
    Пример:
    sentence (incoming) = ["hello", "how",   "are", "you"]
    all_words           = ["hi",    "hello", "I",   "you", "bye", "thank", "cool"] # Собраны из всех patterns, которые хранятся в файле json
    bag                 = [  0 ,       1 ,    0 ,     1 ,    0 ,     0 ,      0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    
    return bag