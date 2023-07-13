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



print(v1)
