#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor

Abordagem para segmentação da base de dados, utilizando o KMeans.
Tentativa de remoção de linhas/manchas na imagem.

Referencia:
    https://scikit-learn.org/stable/modules/clustering.html#k-means
"""

from skimage import io, transform, color, exposure
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os


# Segmentação por K-Means
def kMeans(img):
    w, h, d = img.shape
    image_array = np.reshape(img, (w * h, d))      # Transformando em "vetor" - necessário
    kmeans = KMeans(n_clusters=2, init='random', n_init=1).fit(image_array)
    labels = kmeans.predict(image_array)
    seg1 = np.reshape(labels, (w,h))
    seg1 = (seg1*255).astype('uint8')
    return seg1
    
# Teste manual
def ajusteManual(im, baixo, alto):
    im2 = im.copy()
    im2[im2<=baixo] = 0
    im2[im2>=alto] = 255

    im2[im2>baixo] = 255*(im2[im2>baixo] - baixo)/(alto - baixo)
    return im2.astype('uint8')

# ## Para testar apenas uma imagem:
# path = '/home/victor/01_235.png'
# img = io.imread(path)

# # Resize - Ajuda a melhorar a resposta ao mudar o tamanho de 32x32 => 300x300
# img = transform.resize(img, (300,300))

# img2 = kMeans(img)
# plt.figure()
# plt.imshow(img2, cmap='gray')


## Teste automatizado:
maindir = '/media/victor/1f3aa121-0188-42a3-b7c4-b271c3c0afca/2020 - 1/Processamento de imagens/repositorio_github/letters/example_dataset/'
files = os.listdir(maindir)

## Escolha do método a ser utilizado
choose = 'kmeans'
for dire in files:
    if dire.endswith('.png'):
        print(maindir + dire)
        
        if (choose == 'kmeans'):
            im = io.imread(maindir + dire)
            im3 = transform.resize(im, (300,300))
            im4 = kMeans(im3)
            # im4 = (im4*255)
            plt.subplot(121)         
            plt.imshow(im, cmap="gray")
            plt.title("Imagem original")
            plt.subplot(122)
            plt.imshow(im4, cmap="gray")
            plt.title("Resultado")
            plt.savefig(maindir +'kmean_based_results/'+ dire)
            # io.imsave(maindir +'results/' + dire, im4.astype('uint8'))
        elif (choose == 'ajusteManual'):
            im = io.imread(maindir + dire)
            im2 = transform.resize(im, (300,300)) 
            im3 = color.rgb2gray(im2)
            im3 = (im3*255).astype('uint8')
            im4 = exposure.equalize_hist(im3, nbins=3)
            im4 = color.rgb2gray(im4)
            im4 = (im4*255).astype('uint8')                      
            im4 = ajusteManual(im4,120,130)

            plt.subplot(121)
            plt.imshow(im, cmap="gray")
            plt.title("Imagem original")
            plt.subplot(122)
            plt.imshow(im4, cmap="gray")
            plt.title("Resultado")
            plt.savefig(maindir +'ajusteManual_based_results/'+ dire)
            # io.imsave(maindir +'results/' + dire, im4.astype('uint8'))            