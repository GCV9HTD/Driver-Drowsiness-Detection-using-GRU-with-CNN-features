import numpy as np
import tensorflow as tensorflow
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from keras_squeezenet_tf2 import SqueezeNet
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2 as cv
import os
import time 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


image_height=90
image_width=90

#def expression_to_label(expression):
#	if expression == 'Neutral':
#		return 0
#	if expression == 'Happiness':
#		return 1
#	if expression == 'Fear':
#		return 2
#	if expression == 'Surprise':
#		return 3
#	if expression == 'Sadness':
#		return 4
	
	


#path='sorted_training_5'

#x_train=[]
#y_train=[]

#count=0
#for expression in os.listdir(path):
#	for image in os.listdir(os.path.join(path,expression)):
#		img=cv.imread(os.path.join(path,expression,image),1)
#		img=cv.resize(img,(image_height,image_width))
#		x_train.append(img)
#		y_train.append(expression_to_label(expression))
#		count=count+1
#		print("Loaded: ", count)

#x_test=[]
#y_test=[]
#path1='sorted_validation_5'
#for expression in os.listdir(path1):
#	for image in os.listdir(os.path.join(path1,expression)):
#		img=cv.imread(os.path.join(path1,expression,image),1)
#		img=cv.resize(img,(image_height,image_width))
#		x_test.append(img)
#		y_test.append(expression_to_label(expression))
#		count=count+1
#		print("Loaded: ", count)
#
#x_train=np.array(x_train)
#y_train=np.array(y_train)
#x_test=np.array(x_test)
#y_test=np.array(y_test)

x_train=np.load('90x90/x_train_5.npy', allow_pickle=True)
y_train=np.load('90x90/y_train_5.npy',allow_pickle=True)
x_test=np.load('90x90/x_test_5.npy', allow_pickle=True)
y_test=np.load('90x90/y_test_5.npy',allow_pickle=True)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#np.save('90x90/x_train_5.npy',x_train)
#np.save('90x90/y_train_5.npy',y_train)
#np.save('90x90/x_test_5.npy',x_test)
#np.save('90x90/y_test_5.npy',y_test)

x_train,y_train=shuffle(x_train, y_train, random_state=0)
x_test, y_test=shuffle(x_test, y_test, random_state=0)
print(y_train)

#x_total=np.concatenate((x_train,x_test), axis=0)
#y_total=np.concatenate((y_train, y_test), axis=0)
x_train=x_train/255
x_test=x_test/255

#x_train, x_test, y_train, y_test=train_test_split(x_total, y_total, test_size=0.1)
y_train=to_categorical(y_train, num_classes=5)
y_test=to_categorical(y_test, num_classes=5)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

datagen=ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.15,
    fill_mode="nearest")

inception=tf.keras.applications.InceptionV3(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=(image_height, image_width,3),
    pooling='avg',
    classes=5,
)
x=inception.output
x=tf.keras.layers.Dense(1024,activation='relu')(x)
x=tf.keras.layers.Dense(512,activation='relu')(x)
prediction=tf.keras.layers.Dense(5,activation='softmax')(x)

model=tf.keras.Model(inputs=inception.input, outputs=prediction)

print(model.summary())
print(len(model.layers))

#inception.trainaible=True
#for layer in model.layers[:249]:
#    layer.trainable = False
#for layer in model.layers[249:]:
#    layer.trainable = True
#opt=tf.keras.optimizers.Adam(lr=1e-5)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


#history=model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=256)
h
istory=model.fit_generator(datagen.flow(x_train, y_train, batch_size=256), epochs=100 , validation_data=(x_test, y_test))

#print("Timem Taken: ", time.time()-start)
#model.save('final_model.h5')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(('acc','val_acc'), loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(('loss','val_loss'), loc='upper left')
plt.show()

print("Max validation accuracy: ", max(history.history['val_acc']))
print("Max Training Accuracy: ", max(history.history['acc']))