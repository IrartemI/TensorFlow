# Урок №3 по TensorFlow
import tensorflow as tf 
import numpy as np

# тензор из нулей матрицей 3 на 3
a = tf.zeros((3, 3))

c = tf.ones_like(a) # Тензор С берет за основу  размерность тензора "a"  

d = tf.eye(3) # Тензор 3 на  3 с главной диагональю едениц
dd = tf.eye(3,2) # Тензор 3 на  2 с главной диагональю едениц


# генерация тензоров со случайными значениями

a = tf.random.normal((2,4), 0, 0.1) # Тензор 2 на 4, МО - 0, СКО-0.1

a = tf.random.uniform((2,2), -1, 1) # 2 на 2 от -1 до 1 равномерно

a = tf.random.set_seed(1)
d = tf.random.truncated_normal((1,5), -1, 0.1)




## метематические операции

# Сложение (размерности должны быть одинаковые)
z = tf.constant([1,2,3])
y = tf.constant([9,8,7])

d = tf.add(z,y)


# Вычетание
c = tf.subtract(z,y)

# Деление 
v = tf.divide(z, y)

# Поэлементное умножение 
b = tf.multiply(z,y)


# Внешнее векторное умножение
n = tf.tensordot(z,y, axes = 0)

# Внутреннее векторное умножение
m = tf.tensordot(z,y, axes = 1)



# Перемножение матриц
a2 = tf.constant(tf.range(1,10), shape=(3,3))
b2 = tf.constant(tf.range(5,14), shape=(3,3))

m = tf.matmul(a2,b2)


# Сумма всех элементов тензора
f = tf.reduce_sum(n) 

# Сумма элементов по столбцам 
g = tf.reduce_sum(n, axis=0)

# Сумма элементов по строкам
g = tf.reduce_sum(n, axis=1)


# среднее арифмитическое тензора 
h = tf.reduce_mean(n)

# максивальное значение тензора 
j = tf.reduce_mean(n)

# произведение всех значений тензора 
k = tf.reduce_prod(n)

# нахождение корня 
l = tf.sqrt(tf.cast(n, dtype=tf.float32)) # cast - приводит матрицу к вещественному значению


# возведение в квадрат 
q = tf.square(n)


# нахождение синуса 
w = tf.sin(tf.range(-3.14, 3.14, 1))


print(w)


