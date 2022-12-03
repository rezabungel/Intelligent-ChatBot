import nltk
from nltk.stem.porter import PorterStemmer # Есть различные Стеммеры, мы воспользуемся этим

#nltk.download('punkt') # Используется при первом запуске для работы word_tokenize

# tokenize
def tokenize(sentence):
    return nltk.word_tokenize(sentence)

# stem
stemmer = PorterStemmer()
def stem(word):
    return stemmer.stem(word.lower()) # К слову будет применен Стеммер и слово будет преобразовано в нижний регистр при необходимости

# bag of words
def bag_of_words(tokenized_sentence, all_words):
    pass


messeg = "Example of operation of preprocessing functions or utility functions!"
print(messeg)
messeg = tokenize(messeg)
print(messeg)

print() # new line

words = ["Organize", "organizes", "organizing"]
print(words)
stemmed_words = [stem(w) for w in words]
print(stemmed_words)