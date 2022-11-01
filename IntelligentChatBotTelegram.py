import telebot
from telebot import types

import settings # Импортируем TOKEN

bot = telebot.TeleBot(settings.TOKEN)

# Работа с командами
@bot.message_handler(commands=['start'])
def start(message):
    mess = f'Привет, <u><b>{message.from_user.first_name}</b></u>'
    bot.send_message(message.chat.id, mess, parse_mode='html')# 1 параметр - в какой чат отправляет ответ, 2 параметр - наш ответ, 3 параметр - режим отправки ответов

# Создание кнопок (Кнопка - встраивается в сообщение)
@bot.message_handler(commands=['website'])
def website(message):
    markup = types.InlineKeyboardMarkup() # InlineKeyboardMarkup - класс, который позволяет создавать различные встроенные в сообщения вещи (различные кнопки, изображения  и т.д.)
    markup.add(types.InlineKeyboardButton("Посетить веб сайт pyTelegramBotAPI", url="https://pypi.org/project/pyTelegramBotAPI/"))     # 1 параметр - текст, который написан на кнопке, 2 параметр - URL-адрес
    bot.send_message(message.chat.id, "Сайт pyTelegramBotAPI", reply_markup=markup)

#Создание функциональных кнопок
@bot.message_handler(commands=['help'])
def website(message):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=1) # resize_keyboard = True - масштабирование кнопок под ПК и телефон, row_width=1 - кол-во кнопок в ряду
    website = types.KeyboardButton('photo')  # Параметр - текст на кнопке
    start = types.KeyboardButton('/start')
    markup.add(website, start)
    bot.send_message(message.chat.id, "Какой-то текст", reply_markup=markup)

# Работа с текстом (отслеживание любых сообщений)
@bot.message_handler(content_types=['text'])
def get_user_text(message):
    if message.text == "Hello":
        bot.send_message(message.chat.id, "И тебе привет", parse_mode='html')
    elif message.text == "id":
        bot.send_message(message.chat.id, f"Твой ID, {message.from_user.id}", parse_mode='html')
    elif message.text == "photo":  # Отправка фотографии
        photo = open('something/TestPhoto.jpg', 'rb')# Открытие фотографии стандартными методами Питона. # 'rb' - тип открытия фотографии
        bot.send_photo(message.chat.id, photo)
    else:
        bot.send_message(message.chat.id, "Я тебя не понимаю.", parse_mode='html')

# Работа с документами (боту отправляют фото)
@bot.message_handler(content_types=['photo'])
def get_user_photo(message):
    bot.send_message(message.chat.id, "Это кошка или собака?")

# Запуск бота на постоянное выполнение.
bot.polling(non_stop=True)