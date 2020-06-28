
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger

mnist=tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test)=mnist.load_data()

x_train,x_test=x_train/255.0,x_test/255.0

csv_logger=CSVLogger('training.log')
# you can use: tail -f training.log
        


model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_split=0.2, callbacks=[csv_logger])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        