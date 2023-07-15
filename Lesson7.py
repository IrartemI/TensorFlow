# Урок №5 по TensorFlow
from typing import Any
import tensorflow as tf 
from keras.datasets import mnist
from keras.utils import to_categorical


(x_train, y_train), (x_test, y_test) = mnist.load_data()

# нормируем данные и представим их от 0 до 1
x_train = x_train / 255
x_test = x_test / 255

# Вытяним изображения в оди нединый вектор 
x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28]) #  Длинной в 728
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])


y_train = to_categorical(y_train, 10)

class DenseNN(tf.Module):
    def __init__(self, outputs, activate="relu"):
        super().__init__( )
        self.outputs = outputs
        self.activate = activate
        self.fl_init = False

    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal((x.shape[-1], self.outputs), stddev=0.1, name='w')
            self.b = tf.zeros([self.outputs], dtype = tf.float32, name='b')


            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)

            self.fl_init = True
        y = x @ self.w + self.b

        if self.activate == 'relu':
            return tf.nn.relu(y)
        elif self.activate == 'softmax':
            return tf.nn.softmax(y)
        
        return y



# Создадим 2 слоя

layer_1 = DenseNN(128)
layer_2 = DenseNN(10, activate='softmax')


# пропускание данных через слои 

def model_predict(x):
    y = layer_1(x)
    y = layer_2(y)
    return y


cross_entropy = lambda y_true, y_pred: tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))
# определим оптимизатор для градиентного спуска 
opt = tf.optimizers.Adam(learning_rate=0.001)



# запуск обучения 
BATCH_SIZE = 32
EPOCHS = 10 
TOTAL = x_train.shape[0]  # Размер обучающей выборки 


        # Сделаем разбивку на батчи 

train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)



# Непосредственно само обучение 
for n in range (EPOCHS):
    loss = 0
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape: 
            f_loss = cross_entropy(y_batch, model_predict(x_batch))

        loss += f_loss
        grads = tape.gradient(f_loss, [layer_1.trainable_variables, layer_2.trainable_variables])
        opt.apply_gradients(zip(grads[0], layer_1.trainable_variables))
        opt.apply_gradients(zip(grads[1], layer_2.trainable_variables))

    print(loss.numpy())

# определить качество работы нейронки на тестовой выборки 

y = model_predict(x_test)
y2 =  tf.argmax(y, axis=1)
# acc = len(y_test[y_test == y2])/y_test.shapе[0] * 100
# print(acc)


acc = tf.metrics.Accuracy()
acc.update_state(y_test, y2)
print (acc.result().numpy() * 100)
