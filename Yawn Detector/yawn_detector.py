import tensorflow as tf
from tensorflow.keras.layers import LSTM, Conv1D, Conv2D, Dense, Input, BatchNormalization, Dropout, Bidirectional, TimeDistributed, Flatten, GlobalAveragePooling2D, GRU, SeparableConv1D
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 



def create_windowed_data(x,y, window_size=32):
	dataX,dataY=[],[]
	
	d=0
	d1=0
	for i in range(len(x)-window_size-1):
		dataX.append(x[i:i+window_size])
		count_1=0
		count_0=0
		#dataY.append(y[i:i+window_size])
		for s in y[i:i+window_size]:
			if s==1:
				count_1=count_1+1
			else:
				count_0=count_0+1
		if count_1>=count_0:
			dataY.append(1)
			d=d+1
		else:
			dataY.append(0)
			d1=d1+1
	#convert to numpy and return
	print("count1: ", d)
	print("count0: ",d1)
	return np.array(dataX),np.array(dataY)

x=np.load('x_all_1.npy')
y=np.load('y_all_1.npy')


print("Before")
print(x.shape, y.shape)

x, y=create_windowed_data(x, y)

print("Windowed Data: ", x.shape, y.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y, test_size=0.20, random_state=4)






print("After")
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)




es_callback=tf.keras.callbacks.EarlyStopping(
    monitor='val_acc', patience=10)
checkpoint_filepath='test_dense.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    save_freq='epoch')


model=Sequential([
	SeparableConv1D(8,1	, padding='valid', activation='relu',input_shape=(32,256)),
	Flatten(),
	Dense(8, activation='relu'),
	Dense(1,activation='sigmoid'),
	])
print(model.summary())
opt = tf.keras.optimizers.Adam(lr=1e-3)
model.compile(metrics=['acc'], optimizer=opt, loss='binary_crossentropy')



history=model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test),batch_size=512, callbacks=[es_callback, model_checkpoint_callback])
#print("Evaluation:  ", model.evaluate(x_test,y_test))
model.save('test_dense.h5')
print(history.history.keys())




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
print(history.history['val_acc'])

	