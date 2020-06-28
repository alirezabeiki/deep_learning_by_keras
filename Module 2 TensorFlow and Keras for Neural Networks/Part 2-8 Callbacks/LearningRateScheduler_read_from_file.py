
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import matplotlib.pyplot as plt

%matplotlib inline

mnist=tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test)=mnist.load_data()

x_train,x_test=x_train/255.0,x_test/255.0

def get_learning_rate_from_file(filename, epoch):
    with open(filename, 'r') as f:
        for line in f.readlines():
            line=line.split('#', 1)[0]
            if line:
                par=line.strip().split(':')
                e=int(par[0])
                lr=float(par[1])
                if e <= epoch and lr >0:
                    learning_rate-lr
                else:
                    return learning_rate
                    return learning_rate
        return learning_rate

for i in range(10):
    print('Epoch {}: {}'.format(i, get_learning_rate_from_file('lr.txt', i)))
            
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        