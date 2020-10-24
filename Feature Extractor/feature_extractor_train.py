import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import cv2 as cv
import os
import time 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import math

start=time.time()

def expression_to_label(expression):
	if expression == 'Neutral':
		return 0
	if expression == 'Happiness':
		return 1
	if expression == 'Sadness':
		return 2
	if expression == 'Surprise':
		return 3
	if expression == 'Fear':
		return 4
	
	
image_height=50
image_width=50


#path='refined_data'

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



#x_train=np.array(x_train)
#y_train=np.array(y_train)
#x_test=np.array(x_test)
#y_test=np.array(y_test)

x_train=np.load('refined_data/50x50/x.npy', allow_pickle=True)
y_train=np.load('refined_data/50x50/y.npy',allow_pickle=True)
#x_test=np.load('124x124/x_test_5.npy', allow_pickle=True)
#y_test=np.load('124x124/y_test_5.npy',allow_pickle=True)

print(x_train.shape)
print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)


#np.save('50x50/x.npy',x_train)
#np.save('50x50/y.npy',y_train)
#np.save('124x124/x_test_5.npy',x_test)
#np.save('124x124/y_test_5.npy',y_test)

x_train, x_test, y_train, y_test=train_test_split(x_train, y_train, test_size=0.15)

x_train=x_train/255
x_test=x_test/255
print(y_train)
y_train=to_categorical(y_train, num_classes=5)
y_test=to_categorical(y_test, num_classes=5)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


tensorboard=tf.keras.callbacks.TensorBoard(log_dir='./logs')

def step_decay(epoch):
    initial_lrate = 1e-4
    drop = 0.1
    epochs_drop = 150.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate

lr_callback = tf.keras.callbacks.LearningRateScheduler(step_decay)

es_callback=tf.keras.callbacks.EarlyStopping(
    monitor='val_acc', patience=20)
checkpoint_filepath='refined_data/2nd_part/best_accuracy_our_model_refined_2_50x50.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    save_freq='epoch')

datagen=ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.20,
    height_shift_range=0.20,
    horizontal_flip=True,
    zoom_range=0.15,
    fill_mode='nearest'
    #zca_whitening=True
    )


model=tf.keras.Sequential([
tf.keras.layers.Conv2D(16,(3,3),padding='same',activation='relu',input_shape=(image_height,image_width,3)),
tf.keras.layers.Conv2D(16,(3,3),padding='same',activation='relu'),
tf.keras.layers.MaxPooling2D((2,2),strides=2),
tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu'),
tf.keras.layers.Conv2D(32,(3,3),padding='same',activation='relu'),
tf.keras.layers.MaxPooling2D((2,2),strides=2),
tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
tf.keras.layers.Conv2D(64,(3,3),padding='same',activation='relu'),
tf.keras.layers.MaxPooling2D((2,2),strides=2),
tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'),
tf.keras.layers.Conv2D(128,(3,3),padding='same',activation='relu'),
tf.keras.layers.MaxPooling2D((2,2),strides=2),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(256,activation='relu'),
tf.keras.layers.Dense(5,activation='softmax')
])





print(model.summary())
print(len(model.layers))

opt=tf.keras.optimizers.Adam(lr=1e-4)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['acc'])


#history=model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), batch_size=64)#first batch_size=256

history=model.fit_generator(datagen.flow(x_train, y_train, batch_size=8), steps_per_epoch=len(x_train)/8, epochs=300 ,
 validation_data=(x_test, y_test), callbacks=[model_checkpoint_callback,es_callback])

print("Timem Taken: ", time.time()-start)
model.save('refined_data/2nd_part/final_model_2.h5')
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Accuracy/Model Loss')
plt.ylabel('acc/loss')
plt.xlabel('epochs')
plt.legend(('acc','val_acc', 'loss', 'val_loss'), loc='upper right')
plt.show()


print("Max validation accuracy: ", max(history.history['val_acc']))
print("Max Training Accuracy: ", max(history.history['acc']))