import numpy as np
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from NLTK_utils import tokenize, stem, bag_of_words
from model import NeuralNet

with open('intents.json', 'r') as f:  # Безопасное открытие файла (+ гарантия его закрытия). Режим открытия: r - read mood (открыть файл в режиме чтения)
    intents = json.load(f) # Возвращает JSON объект как словарь (Чтения содержимого JSON файла)

all_words = []
tags = []
xy = []

# Цикл через каждое предложение в нашем intents patterns (Токенизируем все слова из patterns и связываем их с тегом в котреж)
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag) # Добавляем тег в массив
    for pattern in intent['patterns']:
        w = tokenize(pattern) # Токенизируем каждое слово в предложении
        all_words.extend(w) # Добавляем токены в наш массив слов
        xy.append((w, tag)) # Связываем соответсвующий тег и токены в кортеж и добавляем его в массив

# Удаление стоп-слов (знаки пунктуаций) и применяем Стеммера (+преобразование из верхнего регистра в нижний)
stop_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in stop_words]  

all_words = sorted(set(all_words)) # Сортировка и удаление повторных слов
tags = sorted(set(tags)) # Сортировка и удаление повторных тегов (ничего не удалиться, так как все теги уникальны)

# Bag of words
# Creating Training Data
X_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag) # X: bag of words for each pattern_sentence
    label = tags.index(tag)
    y_train.append(label) # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
 
X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # Поддерживание индексации таким образом, чтобы dataset[i] можно было использовать для получения i-й выборки
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index] # Возвращает в виде кортежа

    # Можем вызвать len(dataset), чтобы получить размер
    def __len__(self):
        return self.n_samples

# Hyper-parameters 
num_epochs = 1000
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

#print(input_size, len(all_words))
#print(output_size, tags)
#print(input_size, output_size)

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0) # shuffle - перемешивание, num_workers - мультипоток (в win может быть ошибка, если стоит не 0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Если есть возможность, то обработка будет производиться на GPU, если нет, то на CPU
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # ls - learning rate - скорость обучения

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if ((epoch+1) % 100 == 0): # Вывод информации о каждых 100 эпохах
        print(f"epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}")

print(f"final loss, loss={loss.item():.4f}")
