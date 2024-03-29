import random
import json

import torch

from model import NeuralNet
from NLTK_utils import tokenize, bag_of_words

def Jarvis(messeg): # Функция, выдающая ответ на введенное пользователем сообщение
    messeg = tokenize(messeg)
    X = bag_of_words(messeg, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X) # Предсказываем значение ответов (тегов)
    _, predicted = torch.max(output, dim=1) # Ищем наиболее подходящий ответ (тег - с наибольшим значением)
    tag = tags[predicted.item()] # Получаем фактический тег ответа

    probs = torch.softmax(output, dim=1) # Применяем softmax для получения значения вероятности (уверенности) выборов тегов. (Дальше сделаем проверку на основе вероятности лучшего тега)
    prob = probs[0][predicted.item()] # Берем значение вероятности лучшего тега

    if prob.item() > 0.75: # Если вероятность (уверенность) выбора тега достаточно велика, то мы ищем ответ в JSON файле, иначе говорим, что не понимаем, что ввел пользователь
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                answer = random.choice(intent['responses']) # Случайно выбираем один из возможных ответов
    else:
        answer = "I do not understand..."
    
    return answer



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Если есть возможность, то обработка будет производиться на GPU, если нет, то на CPU

with open('../source/intents.json', 'r') as json_data:  # Безопасное открытие файла (+ гарантия его закрытия). Режим открытия: r - read mood (открыть файл в режиме чтения)
    intents = json.load(json_data) # Возвращает JSON объект как словарь (Чтения содержимого JSON файла)

FILE = "../source/data.pth"
data = torch.load(FILE)

model_state = data["model_state"]
input_size = data["input_size"]
output_size = data["output_size"]
hidden_size = data["hidden_size"]
all_words = data["all_words"]
tags = data["tags"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state) # Теперь модель знает наши изученные параметры
model.eval() # Устанавливаем модель в режим оценки

# Реализация чата в консоли

bot_name = "Jarvis"

# Общение в консоли. Работает, когда запускаем файл с этим кодом. (Если импортируем этот файл, то этот блок кода не выполнится)
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")

    while True:
        sentence = input("You:  ")
        if sentence == "quit":
            break

        answer = Jarvis(sentence)
        print(f"{bot_name}:  {answer}")
