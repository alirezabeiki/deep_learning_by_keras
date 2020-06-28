
import tensorflow as tf
import datetime

mnist=tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test)=mnist.load_data()

x_train,x_test=x_train/255.0,x_test/255.0

class MyCustomCallbacks(tf.keras.callbacks.Callback):
    
    def on_train_begin(self, logs=None):
        print("Training is started!")
        
    def on_epoch_begin(self, epoch, logs=None):
        print("-"*50)
        print("Epoch {} is started.".format(epoch))
        
    def on_train_batch_begin(self, batch, logs=None):
        if batch%100==0:
            print("Training: batch {} begins".format(batch))
            
    def on_test_begin(self, logs=None):
        print("*"*50)
        t=datetime.datetime.now().time()
        print("Evaluation begins at {}".format(t))
        
    def on_test_end(self, logs=None):
        t=datetime.datetime.now().time()
        print("Evaluation finishes at {}".format(t))
      
       
#object
my_callback=MyCustomCallbacks()

model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, validation_split=0.2, callbacks=[my_callback], verbose=0)        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        