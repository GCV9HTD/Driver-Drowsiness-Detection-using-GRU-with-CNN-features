import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tfkerassurgeon.operations import delete_layer
import os
import cv2 as cv

#load our trained model
model=load_model('best_accuracy_our_model_refined_2_50x50.h5')

print(model.summary())

print(len(model.layers))

#get last dense layer
layer_1 = model.layers[14]

#delete last dense layer using keras-surgeon
model=delete_layer(model,layer_1)
print(model.summary())

#save updated model
model.save('feature_extractor_1.h5')

# traverse through folders containing extracted frames from YawDD dataset, load them and add labels
#1 is yawning

x_before=[]
y_before=[]

path='save_train_yawn_f'
for image in os.listdir(path):
	img=cv.imread(os.path.join(path,image),1)
	x_before.append(img)
	y_before.append(1)
path='save_test_yawn_f'
for image in os.listdir(path):
	img=cv.imread(os.path.join(path,image),1)
	x_before.append(img)
	y_before.append(1)
path='save_train_normal_f'
for image in os.listdir(path):
	img=cv.imread(os.path.join(path,image),1)
	x_before.append(img)
	y_before.append(0)
path='save_test_normal_f'
for image in os.listdir(path):
	img=cv.imread(os.path.join(path,image),1)
	x_before.append(img)
	y_before.append(0)

#convert to numpy

x_before=np.array(x_before)
y_before=np.array(y_before)

print(x_before.shape, y_before.shape)

#run model
x_after=model.predict(x_before)
y_after=y_before

print(x_after.shape, y_after.shape)

#save the extracted features and corresponding labels
np.save('x_all_1.npy',x_after)
np.save('y_all_1.npy',y_after)
