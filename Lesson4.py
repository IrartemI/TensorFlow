# Урок №3 по TensorFlow

import tensorflow as tf
import numpy as np


x = tf.Variable (-2.0) # Вещественное число (это важно)

with tf.GradientTape() as tape: # записываем все промежуточные вычисления для y = x**2 и сохраняем их в переменную tape
    y = x**2

df = tape.gradient(y,x)

 




w = tf.Variable(tf.random.normal((3,2))) # Вес. Задан матрицей 3 на 2
b = tf.Variable(tf.zeros(2,dtype=tf.float32)) # Смещение (вектор размерностью 2 элемента)
x = tf.Variable([[-2.0, 1.0, 3.0]]) # Вход - матрица 1 на 3

with tf.GradientTape() as tape:
    y = x @ w + b 
    loss = tf.reduce_mean(y**2) # вычисляем среднее арифметическое


df = tape.gradient(loss, [w,b]) # для loss вычисляем производные по параметру w и b 

print(df[0], df[1], sep="\n")


