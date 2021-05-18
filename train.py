import tensorflow
from tensorflow import keras as ks
import numpy as np
mnist = ks.datasets.mnist
data = [traininp,trainop],[testinp,testop] = mnist.load_data()
traininp = np.concatenate([traininp,testinp],0)
trainop = np.concatenate([trainop,testop],0)
traininp = traininp/255.0
model = ks.models.Sequential()
model.add(ks.layers.Flatten(input_shape=(28,28)))
model.add(ks.layers.Dense(128,activation='relu'))
model.add(ks.layers.Dense(10,activation='softmax',name='Output'))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics = ['accuracy'])
model.fit(traininp,trainop,epochs=50)
ks.models.save_model(model,'./model')