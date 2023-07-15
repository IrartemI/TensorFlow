# Урок №5 по TensorFlow
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt


TOTAL_POINT = 1000

x = tf.random.uniform(shape=[TOTAL_POINT], minval=0, maxval=10)
noise =tf.random.normal(shape=[TOTAL_POINT], stddev=0.2) # Генерация 1000 сточек с нормальным распределением с МО = 0 и СКО = 0.2

k_true = 0.7
b_true = 2.0


y = x * k_true + b_true + noise


plt.scatter(x, y, s=2)
plt.show()


# print(x)



