import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras import Sequential
from tfkerassurgeon.operations import delete_layer
import os
import cv2 as cv

model=load_model('best_accuracy_our_model_refined_2_50x50.h5')

print(model.summary())
print(len(model.layers))
layer_1 = model.layers[14]
model=delete_layer(model,layer_1)
print(model.summary())
model.save('feature_extractor_1.h5')
#load data
#1 is yawning

#x_train_before=[]
#y_train_before=[]
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

x_before=np.array(x_before)
y_before=np.array(y_before)
#x_test_before=np.array(x_test_before)
#y_test_before=np.array(y_test_before)
print(x_before.shape, y_before.shape)
#print(x_before.shape, y_test_before.shape)

#run model


x_after=model.predict(x_before)
#x_test_after=model.predict(x_test_before)
y_after=y_before
#y_test_after=y_test_before

print(x_after.shape, y_after.shape)
#print(x_test_after.shape, y_test_after.shape)
np.save('x_all_1.npy',x_after)
np.save('y_all_1.npy',y_after)
#np.save('x_test.npy', x_test_after)
#np.save('y_test.npy', y_test_after)