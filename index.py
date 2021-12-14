import tensorflow as tf
mnist = tf.keras.datasets.mnist 


(x_train,y_train), (x_test,y_test)  = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# print(x_train[0])

import matplotlib.pyplot as plt 

plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show()
