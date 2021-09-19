#!/usr/bin/env python
# coding: utf-8

# In[15]:



from keras.datasets import cifar10 #importing the cifar10 dataset
import matplotlib.pyplot as plt
(train_X,train_Y),(test_X,test_Y)=cifar10.load_data()
from PIL import Image
import numpy as np
from IPython.display import display, Image


# In[3]:


# Importing the required packages and modules to create our CNN model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


# In[7]:


# normalizing the dataset
train_x=train_X.astype('float32')   # converting the pixel values of the dataset to float type 
test_X=test_X.astype('float32') 
train_X=train_X/255.0
test_X=test_X/255.0


# In[8]:


# one-hot encoding for target classes
train_Y=np_utils.to_categorical(train_Y)
test_Y=np_utils.to_categorical(test_Y)
num_classes=test_Y.shape[1]


# In[9]:


# Creating the  CNN model
model=Sequential()
model.add(Conv2D(32,(3,3),input_shape=(32,32,3),activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(32,(3,3),activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))    # for the output layer we are adding a softmax activation function


# In[12]:


# Compiling the model 
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])


# In[13]:


# Train the model
model.fit(train_X,train_Y,validation_data=(test_X,test_Y),epochs=10,batch_size=32)


# In[29]:


# Make a dictionary to map to the output classes and make predictions from the model

results={
   0:'aeroplane',
   1:'automobile',
   2:'bird',
   3:'cat',
   4:'deer',
   5:'dog',
   6:'frog',
   7:'horse',
   8:'ship',
   9:'truck'
}
from PIL import Image
import numpy as np
file_name=input('enter the file name in jpg format: ')
im=Image.open(file_name)
im.show()
# the input image is required to be in the shape of dataset, i.e (32,32,3)
 
im=im.resize((32,32))
im=np.expand_dims(im,axis=0)
im=np.array(im)
pred=model.predict_classes([im])[0]
print(f'This is a {results[pred]}')
from IPython.display import display, Image
display(Image(filename=file_name))




from PIL import Image
import numpy as np
file_name=input('enter the file name in jpg format: ')
im=Image.open(file_name)
im.show()
im=im.resize((32,32))
im=np.expand_dims(im,axis=0)
im=np.array(im)
pred=model.predict_classes([im])[0]
print(f'This is a {results[pred]}')
from IPython.display import display, Image
display(Image(filename=file_name))



from PIL import Image
import numpy as np
file_name=input('enter the file name in jpg format: ')
im=Image.open(file_name)
im.show()
im=im.resize((32,32))
im=np.expand_dims(im,axis=0)
im=np.array(im)
pred=model.predict_classes([im])[0]
print(f'This is a {results[pred]}')
from IPython.display import display, Image
display(Image(filename=file_name))

