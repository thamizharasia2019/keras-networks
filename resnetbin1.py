#!/usr/bin/env python
# coding: utf-8

# In[1]:

from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os



import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D

from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam

from keras.applications import ResNet50
# from keras.applications import Xception # TensorFlow ONLY

from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
import argparse
import cv2

from keras.callbacks import ModelCheckpoint
from keras.applications.imagenet_utils import preprocess_input


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--dataset", required=True,
	help="path to npz file")

ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
args = vars(ap.parse_args())
# # Using Resnet50 and initialised with weights of imagenet
# ## images in smear 2005 are resized to 224 x224 


INIT_LR = 1e-1
BS = 8
EPOCHS = 5


# loading data

with np.load('./dataset_291119FN/tmp/'+args["dataset"]+'.npz') as data:
    trainX = data['arr_0']
    trainY = data['arr_1']
    testX = data['arr_2']
    testY=  data['arr_3']


print('loaded data')
# initialize an our data augmenter as an "empty" image data generator
#aug = ImageDataGenerator()





# In[2]:


img_height,img_width = 128,128
num_classes = 2
input_shape= (img_height,img_width,3)
base_model=ResNet50(weights='imagenet',include_top=False,input_shape= (img_height,img_width,3)) #imports the mobilenet model and discards the 

# Freeze the layers except the last 4 layers
for layer in base_model.layers[:-3]:
    layer.trainable = False

# Check the trainable status of the individual layers
#for layer in base_model.layers:
#    print(layer, layer.trainable)


# ## Added three dense layers and the last layer is having 7 classes

# In[12]:


x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(num_classes,activation='softmax')(x) #final layer with softmax activation


# ## created new model using base model input and output with 7 classes

# In[13]:


model=Model(inputs=base_model.input,outputs=preds)
#specify the inputs
#specify the outputs
#now a model has been created based on our architecture


# ## Displayed model details

# In[14]:


#model.summary()


# ## Created function to computer F1 SCORE

# In[15]:


from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


# ## Compiled model using Adam optimizer and computed accuracy and f1 score

# In[16]:


print("[INFO] compiling model...")
#opt = SGD(lr=INIT_LR, momentum=0.9, decay=INIT_LR / EPOCHS)


opt = Adam(learning_rate=0.00001)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy',f1])


#model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy',f1])


# ## Applied image data generator with rotation till 90, equalization, flipping, width shift and height shift 

# In[17]:




filepath="AUUG LR 0.00001 data-100 weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


#results=model.fit_generator(
#        train_generator,
#	samples_per_epoch=50,
 #       epochs=50,
 #       validation_data=validation_generator,
 #       callbacks=callbacks_list, 
  #      verbose=1,
   #     validation_steps=20)

print("[INFO] training network for {} epochs...".format(EPOCHS))

H  = model.fit(trainX, trainY,
          batch_size=64,
          epochs=EPOCHS,
          callbacks=callbacks_list, 
          verbose=1,
          validation_data=(testX, testY),
          shuffle=True)







# ## Displaying plot of Accuracy Vs epochs

# In[18]:

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=10)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=le.classes_))
print('Test loss:', predictions [0])
print('Test accuracy:', predictions [1]) 


# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])



import matplotlib
from matplotlib import pyplot as plt
results1=H.history
training_accuracy=results1['accuracy']
val_acc=results1['val_accuracy']
epochs1=range(1,len(training_accuracy)+1)
plt.plot(epochs1,training_accuracy,label='Training Accuracy',marker="*",color='r')
plt.plot(epochs1,val_acc,label='Validation Accuracy',marker="+",color='g')
plt.title('Accuracy Vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')


# ## Displaying plot of Loss Vs epochs

# In[19]:


trainloss=results1['loss']
valloss=results1['val_loss']
epochs1=range(1,len(trainloss)+1)
plt.plot(epochs1,trainloss,label='Training Loss',marker="*",color='r')
plt.plot(epochs1,valloss,label='Validation Loss',marker="+",color='g')
plt.title('Loss Vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')


# ## Displaying plot of F1 score Vs epochs

# In[20]:


trainf1=results1['f1']
valf1=results1['val_f1']
epochs1=range(1,len(trainf1)+1)
plt.plot(epochs1,trainf1,label='Training F1 score',marker="*",color='r')
plt.plot(epochs1,valf1,label='Validation F1 score',marker="+",color='g')
plt.title('F1 score Vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('F1 score')


# In[ ]:




