# Урок №2 по TensorFlow

import tensorflow as tf 
import numpy as np


# Создание обычных тензоров с помощью constant

b = tf.constant([1, 2, 3, 4])
c = tf.constant([[1, 2],
                [3, 4],
                [5, 6]], dtype=tf.float32)


# Тензор который создается из переменных (Variable)
v1 = tf.Variable(-1.2)
v2 = tf.Variable([4, 5, 6, 7], dtype=tf.float32) # Вектор
v3 = tf.Variable(b)


# Замена элементов тензора

v1.assign(0) # Заменили -1.2 на 0
v2.assign([0, 1, 6, 9])

v3.assign_add([1, 1, 1, 1]) # Проссумерает каждое значение 
v1.assign_sub(5)



# списочное индексирование (gather)
x = tf.constant(range(10)) + 5 # Создали тензор 10 чисел на чиная с 5-ки
x_ind = tf.gather(x, [0, 4])  # Выбрали 0 и 4 элемент из тензора (Сформировали новый тензор)


## Изменение формы тензоров по осям

a = tf.constant(range(30))
    # превратим одномерный тензор в двумерный 5 на 6 элементов (reshape)
b = tf.reshape(a,[5, 6])  # создает новое представление данных

# транспонирование 
b_T = tf.transpose(b, perm=[1,0 ]) # perm -  задает что мы меняем строки на столбцы

print(a, b, b_T, sep='\n')
