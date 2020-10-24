import tensorflow as tf
from tensorflow.keras.layers import Dense, GRU, SeparableConv1D
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
 


#create windowed data, such that we input window of 32 in our detector and it outputs 1 or 0 depicting whether yawning detected or not
def create_windowed_data(x,y, window_size=32):
	dataX,dataY=[],[]
	for i in range(len(x)-window_size-1):
		dataX.append(x[i:i+window_size])
		count_1=0
		count_0=0
		for s in y[i:i+window_size]:
			if s==1:
				count_1=count_1+1
			else:
				count_0=count_0+1
		if count_1>=count_0:
			dataY.append(1)
		else:
			dataY.append(0)
	#convert to numpy and return
	return np.array(dataX),np.array(dataY)

#load extracted features from yawdd dataset
x=np.load('x_all_1.npy')
y=np.load('y_all_1.npy')


print("Before")
print(x.shape, y.shape)

#create windows
x, y=create_windowed_data(x, y)

print("Windowed Data: ", x.shape, y.shape)

#split in train and test
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.20, random_state=4)

print("After")
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# apply callabcks to analyze training

#early stopping callback
es_callback=tf.keras.callbacks.EarlyStopping(
    monitor='val_acc', patience=10)

#model checkpoint callback
checkpoint_filepath='test_dense.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    save_freq='epoch')

#define our model
model=Sequential([
	SeparableConv1D(8,1	, padding='valid', activation='relu',input_shape=(32,256)),
	GRU(8),
	Dense(1,activation='sigmoid'),
	])

#print model summary
print(model.summary())

#define optimizer and compile the model
opt = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(metrics=['acc'], optimizer=opt, loss='binary_crossentropy')

#fit the model
history=model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test),batch_size=512, callbacks=[es_callback, model_checkpoint_callback])

#save the model
model.save('test_dense.h5')
print(history.history.keys())

#plot acc, val_acc, loss, val_loss 
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model accuracy/Model Loss')
plt.ylabel('acc/loss')
plt.xlabel('epochs')
plt.legend(('acc','val_acc', 'loss', 'val_loss'), loc='center  right')
plt.show()

print("Max validation accuracy : ", " ", max(history.history['val_acc']))
print("Max Training accuracy : ", " ", max(history.history['acc']))


	
