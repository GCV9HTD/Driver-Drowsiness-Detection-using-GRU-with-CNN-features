import numpy as np
import tensorflow as tensorflow
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from keras_squeezenet_tf2 import SqueezeNet
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImgaeDataGenerator


image_height=60
image_width=60

datagen=ImgaeDataGenerator(rescale=1.0/255.0)

train_it=datagen.flow_from_directory('sorted_training', class_mode='categorical',batch_size=1024)
validate_it=datagen.flow_from_directory('sorted_validation', class_mode='categorical', batch_size=1024)

class_weights=class_weight.compute_class_weight('balanced', np.unique(train_it.classes),train_it.classes)
class_weight_dict =dict(enumerate(class_weights))

checkpoint_filepath='best_accuracy.h5'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_acc',
    mode='max',
    save_best_only=True,
    save_freq='epoch')

print("Intitalizing SqueezeNet Model")
squeezenet = SqueezeNet(include_top=False, weights=None,
               input_tensor=None, input_shape=(image_height,image_width,3),
               pooling=None,
               classes=8)
x=squeezenet.output
x=tf.keras.layers.Flatten()(x)
x=tf.keras.layers.Dense(1024,activation='relu')(x)
x=tf.keras.layers.BatchNormalization()(x)
x=tf.keras.layers.Dropout(0.2)(x)
x=tf.keras.layers.Dense(512,activation='relu')(x)
output=tf.keras.layers.Dense(8,activation='sigmoid')(x)

model=tf.keras.Model(inputs=squeezenet.input, outputs=output)

print(model.summary())
print(len(model.layers))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#history=model.fit(train_it,epochs=10, batch_size=1024, validation_data=validate_it, class_weight=class_weight_dict, callbacks=[model_checkpoint_callback])
history=model.fit_generator(train_it, steps_per_epoch=len(train_it)/1024, validation_data=validate_it, vaidation_steps=len(validate_it)/1024, class_weight=class_weight_dict, callbacks=[model_checkpoint_callback])

print(history.history.keys())


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
