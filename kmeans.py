#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor

Abordagem para segmentação da base de dados, utilizando o KMeans.
Tentativa de remoção de linhas/manchas na imagem.

Referencia:
    https://scikit-learn.org/stable/modules/clustering.html#k-means
"""

from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# path = '/media/victor/1f3aa121-0188-42a3-b7c4-b271c3c0afca/2020 - 1/Processamento de imagens/datasets - processamento imagens/images_segmentation/2/04_122.png'
path = '/home/victor/01_235.png'
img = io.imread(path)

# Resize - Ajuda a melhorar a resposta ao mudar o tamanho de 32x32 => 300x300
img= transform.resize(img, (300,300))

fig = plt.figure()
plt.subplot(121)
plt.tight_layout()
plt.imshow(img)


# Segmentação por K-Means
w, h, d = img.shape
image_array = np.reshape(img, (w * h, d))      # Transformando em "vetor" - necessário
kmeans = KMeans(n_clusters=2, init='random', n_init=1).fit(image_array)
labels = kmeans.predict(image_array)
seg1 = np.reshape(labels, (w,h))

plt.subplot(122)
plt.imshow(seg1, cmap="gray")
plt.tight_layout()
plt.colorbar()
