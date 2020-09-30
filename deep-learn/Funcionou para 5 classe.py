import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import tensorflow as tf

from skimage import io, transform
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np


maindir_treino = 'C:\\Users\\Pigozzo\\OneDrive\\UTFPR\\2020-1\\Processamento de Imagen\\T2\\Data2\\treino'
maindir_teste = 'C:\\Users\\Pigozzo\\OneDrive\\UTFPR\\2020-1\\Processamento de Imagen\\T2\\Data2\\teste'
lista_treino = []
lista_teste = []
train_images = np.zeros((387*5, 32, 32, 4))
test_images = np.zeros((43*5, 32, 32, 4))
train_labels = np.zeros((387*5, 1))
test_labels = np.zeros((43*5, 1))


for dire in os.listdir(maindir_treino):
    if dire.endswith('.png'):
        lista_treino.append(dire)
        
for dire in os.listdir(maindir_teste):
    if dire.endswith('.png'):
        lista_teste.append(dire)
        
    

id_treina = 0
for i in range(len(lista_treino)):
        im = io.imread(maindir_treino + '/' + lista_treino[i])
        im2 = transform.resize(im,(32,32,4),cval=0)
        train_images[id_treina,:,:,:] = im2
        if(lista_treino[i][1] == '1'):
            train_labels[id_treina] = 0
        if(lista_treino[i][1] == '3'):
            train_labels[id_treina] = 1
        if(lista_treino[i][1] == '9'):
            train_labels[id_treina] = 2
        if(lista_treino[i][1] == '6'):
            train_labels[id_treina] = 3
        if(lista_treino[i][1] == '2'):
            train_labels[id_treina] = 4
        id_treina += 1
        
        
id_treina = 0
for i in range(len(lista_teste)):
        im = io.imread(maindir_teste + '/' + lista_teste[i])
        im2 = transform.resize(im,(32,32,4),cval=0)
        test_images[id_treina,:,:,:] = im2
        if(lista_teste[i][1] == '1'):
            test_labels[id_treina] = 0
        if(lista_teste[i][1] == '3'):
            test_labels[id_treina] = 1
        if(lista_teste[i][1] == '9'):
            test_labels[id_treina] = 2
        if(lista_teste[i][1] == '6'):
            test_labels[id_treina] = 3
        if(lista_teste[i][1] == '2'):
            test_labels[id_treina] = 4
        id_treina += 1
        
# Normalize pixel values to be between 0 and 1
# train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 4)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5))


print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=50, 
                    validation_data=(test_images, test_labels))

plt.figure()
plt.plot(history.history['acc'], label='accuracy')
plt.plot(history.history['val_acc'], label = 'val_accurancy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
        
