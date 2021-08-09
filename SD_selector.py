#Импорт библиотек, необходимых для работы кода.

import matplotlib.pyplot as plt  # Для вывода графиков
import numpy as np  # Для работы с данными
import pandas as pd  # Для работы с таблицами
from sklearn.model_selection import train_test_split  # Для разделения выборки на тестовую и обучающую
from sklearn.preprocessing import LabelEncoder  # Метод кодирования тестовых лейблов
from tensorflow.keras import utils  # Для работы с категориальными данными
from tensorflow.keras.layers import Dense, Dropout  # Слои для сети
from tensorflow.keras.models import Sequential  # Полносвязная модель
from tensorflow.keras.preprocessing.text import Tokenizer  # Методы для работы с текстами и преобразования их в последовательности

file_dir = 'c:/content/drive/MyDrive/Базы/заявки РИГ.xlsx' #Файл для обучения сети
df = pd.read_excel(file_dir) #загружаем данные в dataframe
df.head(5)    #посмотрим на содержимое

df_sd = df.copy() #Создание копии, для опытов
df_sd.dropna(axis=1, how= 'any', inplace= True) #Удалим стоблцы с пустыми значениями

print(df_sd.shape) #Выведем форму таблицы

print(df_sd.values[11]) #Выведем пример из таблицы

#Посмотрим какие услуги, и сколько по ним обращений.
for cl in df_sd['Услуга'].unique(): #Проходим по всем классам
  print('Количество записей по услуге: ', cl, ': ', df_sd[df_sd.Услуга == cl].shape[0])

texts = df_sd['Тема'].values #Извлекаем данные всех текстов из столбца Тема
classes = list(df_sd['Услуга'].values) #Извлекаем соответствующие им значения классов (лейблов) столбца Услуга
maxWordsCount = 60000 #Зададим максимальное количество слов/индексов, учитываемое при обучении текстов

print(df_sd['Услуга'].unique()) #Выводим все уникальные значения классов
uClasses = df_sd['Услуга'].unique() #Массив Услуг
nClasses = df_sd['Услуга'].nunique()+1  #Задаём количество классов, обращаясь к столбцу category и оставляя уникальные значения
print(nClasses) #Посмотрим на количество классов

'''
Преобразовываем текстовые данные в числовые/векторные для обучения нейросетью
Для этого воспользуемся встроенной в Keras функцией Tokenizer для разбиения текста и превращения в матрицу числовых значений
num_words=maxWordsCount - определяем макс.кол-во слов/индексов, учитываемое при обучении текстов
filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n' - избавляемся от ненужных символов
lower=True - приводим слова к нижнему регистру
split=' ' - разделяем слова по пробелу
char_level=False - просим токенайзер не удалять однобуквенные слова
'''
tokenizer = Tokenizer(num_words=maxWordsCount, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ', oov_token='unknown', char_level=False)

tokenizer.fit_on_texts(texts) #"Скармливаем" наши тексты, т.е. даём в обработку методу, который соберет словарь частотности

#Формируем матрицу индексов по принципу Bag of Words
xAll = tokenizer.texts_to_matrix(texts) #Каждое слово из текста нашло свой индекс в векторе длиной maxWordsCount и отметилось в нем единичкой
print(xAll.shape)  #Посмотрим на форму текстов
print(xAll[0, :20])#И отдельно на фрагмент начала вектора

print(tokenizer.word_index.items()) #Вытаскиваем индексы слов для просмотра

#Преобразовываем категории в векторы
encoder = LabelEncoder() # Вызываем метод кодирования тестовых лейблов из библиотеки sklearn
encoder.fit(classes) # Подгружаем в него категории из нашей базы
classesEncoded = encoder.transform(classes) # Кодируем категории
print(encoder.classes_)
print(classesEncoded.shape)
print(classesEncoded[:10])

yAll = utils.to_categorical(classesEncoded, nClasses) # И выводим каждый лейбл в виде вектора длиной 22, с 1кой в позиции соответствующего класса и нулями
print(yAll.shape) # Посмотрим на форму лейблов категорий
print(yAll[0]) # И отдельно на первую строку

# разбиваем все данные на обучающую и тестовую выборки с помощью метода train_test_split из библиотеки sklearn
xTrain, xVal, yTrain, yVal = train_test_split(xAll, yAll, test_size=0.2)
print(xTrain.shape) #посмотрим на форму текстов из обучающей выборки
print(yTrain.shape) #и на форму соответсвующих им классов

# для Эмбэддинга представляем текст в виде последовательности индексов слов,
# каждый из которых затем преобразуем в многомерный вектор
Sequences = tokenizer.texts_to_sequences(texts) #разбиваем текст на последовательность индексов
npSequences = np.array(Sequences)                #переведём в массив numpy
xTrainE, xValE, yTrainE, yValE = train_test_split(npSequences, yAll, test_size=0.2)#сформируем новую выборку
print(xTrainE.shape) #посмотрим на форму текстов из обучающей выборки
print(yTrain.shape)  #и на форму соответствующих им классов

#Полносвязная сеть, распознавания ticket`s
model01 = Sequential()
#Входной полносвязный слой
model01.add(Dense(100, input_dim=maxWordsCount, activation="relu"))
#Слой регуляризации Dropout
model01.add(Dropout(0.3))
#Второй полносвязный слой
model01.add(Dense(100, activation='relu'))
#Слой регуляризации Dropout
model01.add(Dropout(0.3))
#Третий полносвязный слой
model01.add(Dense(100, activation='relu'))
#Слой регуляризации Dropout
model01.add(Dropout(0.3))
#Выходной полносвязный слой
model01.add(Dense(nClasses, activation='softmax'))


model01.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Обучаем сеть на выборке
SD_history = model01.fit(xTrain,
                    yTrain,
                    epochs=20,
                    batch_size=128,
                    validation_data=(xVal, yVal))

plt.plot(SD_history.history['accuracy'],
         label='Доля верных ответов на обучающем наборе')
plt.plot(SD_history.history['val_accuracy'],
         label='Доля верных ответов на проверочном наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()

#Функция проверки распознавания услуги
def ticket(eText):
  Example = tokenizer.texts_to_sequences([eText]) #разбиваем текст на последовательность индексов
  npExpl = tokenizer.sequences_to_matrix(Example) #преобразовываем проверочную строку в BoW
  prediction = model01.predict(npExpl)  #Строим выходной вектор значений.
  sNum = np.argmax(prediction) #Определяем максимальное значение prediction
  print(encoder.classes_[sNum]) #Находим в классификаторе соответсвующую Услугу.

ticket(input('Опишите проблему: '))
