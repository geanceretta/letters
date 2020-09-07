#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor

Abordagem para segmentação da base de dados, tentantiva com hysteresis threshold

Referencia:
    https://scikit-image.org/docs/dev/auto_examples/filters/plot_hysteresis.html
"""

import matplotlib.pyplot as plt
from skimage import filters, io, transform, color

path = '/media/victor/1f3aa121-0188-42a3-b7c4-b271c3c0afca/2020 - 1/Processamento de imagens/datasets - processamento imagens/classification-letters/letters/01_14.png'
image = io.imread(path)
image = color.rgb2gray(image)                        # dtype('float64')
image = (image * 255).astype('uint8')                # Converter para uint8
image_res = transform.resize(image, (300,300))
image_res = (image * 255).astype('uint8')            


## Filtro Prewitt
edges_p = filters.sobel(image_res)

low_p = 0.05
high_p = 0.6

lowt_p = (edges_p > low_p).astype(int)
hight_p = (edges_p > high_p).astype(int)
hyst_p = filters.apply_hysteresis_threshold(edges_p, low_p, high_p)

## Filtro Sobel
edges_s = filters.sobel(image_res)

low_s = 0.05
high_s = 0.6

lowt_s = (edges_s > low_s).astype(int)
hight_s = (edges_s > high_s).astype(int)
hyst_s = filters.apply_hysteresis_threshold(edges_s, low_s, high_s)


fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Imagem original')

ax[1].imshow(lowt_p, cmap='gray')
ax[1].set_title('Prewitt')

ax[2].imshow(lowt_s, cmap='gray')
ax[2].set_title('Sobel')


plt.tight_layout()
plt.show()
# plt.savefig('example')
