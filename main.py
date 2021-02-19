# импортируем модуль для работы с векторами
import numpy as np
# импортируем модуль для работы с датафреймами
import pandas as pd
# импортируем модуль для работы со временем
import time
# импортируем модуль для работы со средствами операционной системы
import os
# импортируем из класса models метод Sequential для инициализации модели
from keras.models import Sequential
# импортируем из класса layers метод Dense для создания слоя нейросети
from keras.layers import Dense
# импортируем из класса utils метод plot_model для создания графической интерпретации модели
from keras.utils import plot_model

class Database:
    # конструктор класса
    def __init__(self, path):
        # в конструкторе класса инициализируем базу данных по заданному пути
        self.df = self.open_db(path)
    # функция считывания заранее подготовленной базы данных
    def open_db(self, path):
        # считывание базы данных и сохранение ее в объекте класса
        # для удобства чтения и дальнейшего масштабирования проекта
        # функция будет возвращать значение
        return pd.read_csv(path)
    # функция конвертирования текстовых лейблов в численные
    def convert_labels(self):
        # заменяем в базе данных лейблы male и female
        # на соответствующие маркеры 0 и 1
        # для того, чтобы компьютер смог однозначно работать с числами
        self.df.loc[self.df['label']=='male', 'label'] = 0
        self.df.loc[self.df['label']=='female', 'label'] = 1
    # функция, разделяющая данные на тренировочные и конечные тестовые
    def split_data(self, df):
        # разделяем базу данных с коэффициентом 80/20
        train_input = df.sample(frac=0.8, random_state=200)
        test_input = df.drop(train_input.index)
        # возвращаем данные
        return train_input, test_input
    # функция, которая распределяет данные на предикторы и определители
    def get_data(self):
        # для существующей базы данных используем метод объекта для изменения значений лейблов
        self.convert_labels()
        # получаем значения разделенных входных данных
        train_input, test_input = self.split_data(self.df)
        # для генерации "случайных" чисел определенной последовательности
        np.random.seed(1)
        self.test_X = np.array(test_input.iloc[:,0:20])
        self.test_Y = np.array(test_input.iloc[:,20])
        self.train_X = np.array(train_input.iloc[:,0:20])
        self.train_Y = np.array(train_input.iloc[:,20])

class Model():
    # конструктор класса
    def __init__(self, path):
        # инициализируем базу данных через класс объекта через прежде написанный класс Database
        self.db = Database(path)
        # получаем для объекта класса db значения тренировочных и тестовых данных
        self.db.get_data()
    # вызов модели
    def call(self):
        # инициализируем объект класса
        self.model = Sequential()
        # создаем 1 скрытый слой с 4 нейронами и нелинейной функцией активации relu
        self.model.add(Dense(4, input_dim=20, activation='relu'))
        # создаем 2 скрытый слой с 2 нейронами и нелинейной функцией активации relu
        self.model.add(Dense(2, activation='relu'))
        # создаем 3 слой с 1 нейроном и нелинейной сигмоидной функцией активации
        self.model.add(Dense(1, activation='sigmoid'))
        # компилируем модель с функцией потери "Бинарная кроссэнтропия" и функцией оптимизации "Адам"
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    def train(self, epochs=100, batch_size=16):
        # сохраняем время начала тренировки в формате unix-time
        start_time = time.time()
        # подача предикторов и определителей
        # epochs --> количество итераций batch_size --> определенный набор данных, на основе которых предсказывается следующее число
        # итерации взятия набора данных происходят до тех пор, пока они не закончатся в базе данных, тем самым настраивая веса синапсов, 
        # после чего начинается следующая эпоха (итерация).
        # переменная validation_split принимает коэффициент, то есть часть, 
        # которую возьмет модель для валидации процесса и поитеративной сверке тренировочных данных для оптимальной настройки модели нейросети
        self.model.fit(self.db.train_X, self.db.train_Y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        # подсчет затраченного времени на тренировку
        print("FIT: {} seconds".format(time.time() - start_time))
    # функция, проводящая сверку с конечными тестовыми данными, выводя точность предсказаний модели
    def test(self):
        results = self.model.evaluate(self.db.test_X, self.db.test_Y, batch_size=128)
        print('TEST RESULT: test_loss {}, test_accuracy {}'.format(results[0], results[1]))
    def info(self):
        # вывод значений параметров (количество синапсов)
        self.model.summary()
        plot_model(self.model, 'keras_model.png', show_shapes=True)
        
def main():
    # Создаем объект класса Model, который использует класс Database
    # Это дает возможность задать путь до базы данных напрямую
    model = Model('db_voices.csv')

    # создание и компиляция модели
    model.call()

    # запуск тренировки модели нейронной сети с заданными параметрами
    model.train(epochs=200, batch_size=16)

    # выполнение предсказаний на основе обученной модели и вывод точности
    model.test()

    # вывод характеристик модели:
    # 1. сведения о слоях
    # 2. количество параметров (весов синапсов)
    # 3. сохранение графического вида модели
    model.info()

# запуск алгоритма
if __name__ == "__main__":
    try:
        main()
    except:
        raise