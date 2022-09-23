import tensorflow as tf
from tensorflow.keras import layers

def get_model():
        model = tf.keras.Sequential()
        model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='Same', 
                        activation=tf.nn.relu, input_shape = (28,28,1)))
        model.add(layers.Conv2D(filters=32, kernel_size=(5,5), padding='Same', 
                        activation=tf.nn.relu))
        model.add(layers.MaxPool2D(pool_size=(2,2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='Same', 
                        activation=tf.nn.relu, input_shape = (28,28,1)))
        model.add(layers.Conv2D(filters=64, kernel_size=(3,3), padding='Same', 
                        activation=tf.nn.relu))
        model.add(layers.MaxPool2D(pool_size=(2,2),strides=(2,2)))
        model.add(layers.Dropout(0.25))

        model.add(layers.Flatten())
        model.add(layers.Dense(256,activation=tf.nn.relu))
        model.add(layers.Dropout(0.25))
        model.add(layers.Dense(10,activation=tf.nn.softmax))
        return model

