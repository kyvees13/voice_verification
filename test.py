import numpy as np
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model

#class Database:
#    def __init__(self, path):
#        self.path = path
#
#    def open_db(self):
#        self.df = pd.read_csv(self.path)
#        return self.df
#    
#    def get_data(self):


# считывание базы данных
df = pd.read_csv("voice.csv")

# заменяем в базе данных тэги male и female
# на соответствующие маркеры 0 и 1
df.loc[df['label']=='male', 'label'] = 0
df.loc[df['label']=='female', 'label'] = 1

# разделяем базу данных на тренировочные данные
# с коэффициентом 80/20
train_input = df.sample(frac=0.8, random_state=200)
# тестовые данные
test_input = df.drop(train_input.index)

# для генерации "случайных" чисел определенной последовательности
np.random.seed(1)
train_X = np.array(train_input.iloc[:,0:20])
train_Y = np.array(train_input.iloc[:,20])
test_X = np.array(test_input.iloc[:,0:20])
test_Y = np.array(test_input.iloc[:,0:20])

model = Sequential()
model.add(Dense(4, input_dim=20, activation='relu'))
model.add(Dense(2, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# компиляция модели с определенными параметрами функции потери и оптимизации сети
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# сохраняем время начала тренировки в формате unix-time
start_time = time.time()

# подача предикторов и определителей
# 200 итераций по 10 пачек входных данных для подсчета
model.fit(train_X, train_Y, epochs=200, batch_size=10)

# вывод значений параметров ( количество весов (синапсов) )
model.summary()

# сохранение графической модели сети
plot_model(model, 'keras_model.png', show_shapes=True)

# подсчет затраченного времени
print("seconds to fit:", time.time() - start_time)

# инициализация предсказанных значений
predictions = model.predict(test_X)

rounded = [round(x[0]) for x in predictions]
print(rounded[:40])

# вычисление точности сети
diff = np.array(test_input['label'].values) - rounded

values, counts = np.unique(1, return_counts=True)
print("Accuracy: {}".format( (1-(counts/len(diff)))*100) )

#def main():
#    24

#if __name__ == "__main__":
#    try:
#        main()
#    except:
#        raise