
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

%matplotlib inline

mnist=tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test)=mnist.load_data()

x_train,x_test=x_train/255.0,x_test/255.0

def scheduler(epoch):
    if epoch < 2:
        return 0.001
    elif epoch < 6:
        return 0.0001
    else:
        return 0.00001
    
learning_rate_scheduler=LearningRateScheduler(scheduler, verbose=1)
        
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history=model.fit(x_train, 
                  y_train, 
                  epochs=10, 
                  validation_split=0.2, 
                  callbacks=[learning_rate_scheduler])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        