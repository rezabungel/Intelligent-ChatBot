import telebot
from telebot import types

from console_chat_with_Jarvis import Jarvis, bot_name
import settings # Импортируем TOKEN

bot = telebot.TeleBot(settings.TOKEN)

# Команда "/start" - выведет небольшое знакомство -> Работа с командами
@bot.message_handler(commands=['start'])
def start(message):
    welcome = f'Привет, <b>{message.from_user.first_name}</b>.\
            \nМеня зовут <b>{bot_name}</b>. Мой создатель <b>Святченко Артём, студент РТУ МИРЭА</b>.\
            \n\nБольше информации обо мне можно узнать, введя команду "/info" или воспользовавшись соответствующей функциональной кнопкой.\
            \n\nА чтобы узнать больше о моем создателе, введите команду "/contacts" или воспользуйтесь соответствующей функциональной кнопкой.\
            \n\nЕсли вдруг функциональные кнопки не появились, то их нужно создать командой "/buttons".'
    bot.send_message(message.chat.id, welcome, parse_mode='html') # 1 параметр - в какой чат отправляет ответ, 2 параметр - наш ответ, 3 параметр - режим отправки ответов

# Команда "/buttons" - создаст функциональные кнопки. (При первом запуске бота, обычно этих кнопок нет) -> Работа с командами + Создание функциональных кнопок
@bot.message_handler(commands=['buttons'])
def create_function_buttons(message):
   markup = types.ReplyKeyboardMarkup(resize_keyboard=True, row_width=3) # resize_keyboard = True - масштабирование кнопок под ПК и телефон, row_width=1 - кол-во кнопок в ряду
   item_start = types.KeyboardButton('/start') # Параметр - текст на кнопке
   item_info = types.KeyboardButton('/info')
   item_contacts = types.KeyboardButton('/contacts')
   item_documentation = types.KeyboardButton('/documentation')
   item_photo = types.KeyboardButton('/myphoto')
   item_buttons = types.KeyboardButton('/buttons')
   
   markup.add(item_start, item_info, item_contacts, item_documentation, item_photo, item_buttons)
   bot.send_message(message.chat.id, "Функциональные кнопки готовы к работе.", reply_markup=markup)

# Команда "/info" - выведет больше информации о боте -> Работа с командами + Создание кнопок, встроенных в сообщение
@bot.message_handler(commands=['info'])
def info_about_Jarvis (message):
    answer = f'Меня зовут <b>{bot_name}</b>. Мой создатель <b>Святченко Артём</b> дал мне это имя будучи вдохновлен искусственным интеллектом, созданным <b>Тони Старком, он же Железный человек, из киновселенной Марвел</b>.\
        \n\nЧто я умею?\
        \n- Я умею общаться. Для этого всего-то нужно написать что-то на английском и разговор завяжется сам собой. Это моя главная задача, для этого я был создан.\
        \n- Еще я знаю ссылки на документации различных python библиотек. Чтобы я вам их показал, нужно воспользоваться командой "/documentation" или соответствующей функциональной кнопкой.\
        \n- Также я могу посмотреть на вашу фотографию и показать свою. Чтобы посмотреть на меня, нужно воспользоваться командой "/myphoto" или соответствующей функциональной кнопкой, а чтобы я посмотрел на ваше фото, просто пришлите мне ее.\
        \n\nМой создатель относит меня к ботам типа small talk.\
        \nSmall talk — это непринужденный разговор на отвлеченные темы, например разговор о погоде.\
        \nФактически я являюсь решением одной из задач NLP (Natural Language Processing).\
        \n\nРаботаю я под управлением нейронной сети с прямой связью и двумя скрытыми слоями.\
        \nКонвейер предварительной обработки (NLP preprocessing pipeline) следующий: string(messeg) -> tokenize -> lower+stem -> exclude punctuation characters(stop words) -> bag of word->getting a one-hot vector.\
        \n\nМою реализацию можно найти в GitHub репозитории.'
    
    markup = types.InlineKeyboardMarkup() # InlineKeyboardMarkup - класс, который позволяет создавать различные встроенные в сообщения вещи (различные кнопки, изображения и т.д.)
    markup.add(types.InlineKeyboardButton("GitHub repository", url="https://github.com/rezabungel/Intelligent-ChatBot")) # 1 параметр - текст, который написан на кнопке, 2 параметр - URL-адрес

    bot.send_message(message.chat.id, answer, reply_markup=markup, parse_mode='html')

# Команда "/contacts" - выведет больше информации о создателе (Святченко Артём) -> Работа с командами + Создание кнопок, встроенных в сообщение
@bot.message_handler(commands=['contacts'])
def info_about_creator (message):
    answer = 'Святченко Артём, студент РТУ МИРЭА.\
            \nСсылка на мой GitHub профиль.'
    
    markup = types.InlineKeyboardMarkup() # InlineKeyboardMarkup - класс, который позволяет создавать различные встроенные в сообщения вещи (различные кнопки, изображения и т.д.)
    markup.add(types.InlineKeyboardButton("GitHub profile", url="https://github.com/rezabungel")) # 1 параметр - текст, который написан на кнопке, 2 параметр - URL-адрес
    
    bot.send_message(message.chat.id, answer, reply_markup=markup)

# Команда "/documentation" - выдаст названия и ссылки на документации python библиотек -> Работа с командами + Создание кнопок, встроенных в сообщение
@bot.message_handler(commands=['documentation'])
def documentation(message):
    python_libraries = [["Scikit-learn", "https://scikit-learn.org/stable/#"], ["NLTK (Natural Language Toolkit)", "https://www.nltk.org"], ["NumPy", "https://numpy.org/doc/stable/"], ["Pandas", "https://pandas.pydata.org/docs/"],
                        ["Pytorch", "https://pytorch.org/docs/stable/index.html"], ["Matplotlib", "https://matplotlib.org/stable/api/index.html"], ["Jupyter", "https://docs.jupyter.org/en/latest/"], ["pyTelegramBotAPI", "https://pytba.readthedocs.io/en/latest/index.html"]]

    for library in python_libraries:
        markup = types.InlineKeyboardMarkup() # InlineKeyboardMarkup - класс, который позволяет создавать различные встроенные в сообщения вещи (различные кнопки, изображения и т.д.)
        markup.add(types.InlineKeyboardButton(library[0], url=library[1])) # 1 параметр - текст, который написан на кнопке, 2 параметр - URL-адрес
        bot.send_message(message.chat.id, f"Документация {library[0]}", reply_markup=markup)

# Команда "/myphoto" - выведет фотографию Jarvis и IronMan -> Работа с командами + Вывод фотографий
@bot.message_handler(commands=['myphoto'])
def show_photo(message):
    #Отправка фотографии 1
    bot.send_message(message.chat.id, "Так я вижу себя:", parse_mode='html')
    photo = open('../source/Jarvis.png', 'rb') # Открытие фотографии стандартными методами Питона. # 'rb' - тип открытия фотографии
    bot.send_photo(message.chat.id, photo)
    #Отправка фотографии 2
    bot.send_message(message.chat.id, "Может быть вы ходите посмотреть еще на Железного человека:", parse_mode='html')
    photo = open('../source/IronMan.jpg', 'rb')
    bot.send_photo(message.chat.id, photo)

# Работа с текстом (отслеживание любых текстовых сообщений)
@bot.message_handler(content_types=['text'])
def get_user_text(message):
    Jarvis_answer = Jarvis(message.text)
    bot.send_message(message.chat.id, Jarvis_answer, parse_mode='html')

    # Вывод сообщений пользователей и ответов бота на них в консоль
    # Вывод имени, логина, id и сообщения пользователя в консоль. (Имя, логин, id - позволяют определить конкретного пользователя, который вводил сообщение)
    print(f'Name:{message.from_user.first_name} Username:{message.from_user.username} id:{message.from_user.id} message_from_user:   {message.text}')
    # Вывод ответа бота в консоль 
    print(f"Jarvis' answer:   {Jarvis_answer}\n")

# Работа с документами (боту отправляют фото)
@bot.message_handler(content_types=['photo'])
def photo_detected(message):
    answer = 'Обнаружил, что вы отправили мне фотографию, но я умею только разговаривать на английском языке.\
        \n\nКстати, если хотите посмотреть на меня, введите команду "/myphoto" или воспользуйтесь соответствующей функциональной кнопкой.'
    bot.send_message(message.chat.id, answer, parse_mode='html')

# Запуск бота на постоянное выполнение.
bot.polling(non_stop=True)