#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor

Abordagem para segmentação da base de dados, utilizando diferentes filtros associados a um resize para melhoria de resultado

Referencia:
    https://scikit-image.org/docs/stable/api/skimage.filters.html
"""

import matplotlib.pyplot as plt
from skimage import filters, io, color, transform

path = 'letters-dataset/letters/01_72.png'

image = io.imread(path)
image = color.rgb2gray(image)            # dtype('float64')
image = (image * 255).astype('uint8')    # Converter para uint8

# Resize da imagem
image_res= transform.resize(image, (300,300))
# image_res = (image_res * 255).astype('uint8')    # Converter para uint8

# Filtros sem resize
edge_roberts = filters.roberts(image)
edge_sobel = filters.sobel(image)
edge_scharr = filters.scharr(image)
edge_prewitt = filters.prewitt(image)
edge_sato_false = filters.sato(image, sigmas=range(1,6,1), black_ridges=False)
edge_sato_true= filters.sato(image, sigmas=range(1,6,1), black_ridges=True)

# erro - perguntar Pablo
edge_threshold_otsu = filters.threshold_otsu(image)


# Filtros com resize
edge_roberts_res = filters.roberts(image_res)
edge_sobel_res = filters.sobel(image_res)
edge_scharr_res = filters.scharr(image_res)
edge_prewitt_res = filters.prewitt(image_res)
edge_sato_false_res = filters.sato(image_res, sigmas=range(1,6,1), black_ridges=False)
edge_sato_true_res = filters.sato(image_res, sigmas=range(1,6,1), black_ridges=True)


# Comparação da melhoria com resize
plt.figure()
plt.imshow(image, cmap='gray')
plt.title('Imagem - Greyscale')
# plt.savefig('Imagem - Greyscale')

plt.figure()
plt.imshow(image_res, cmap='gray')
plt.title('Imagem - Greyscale - Resized')
# plt.savefig('Imagem - Greyscale - Resized')

# Resultado dos filtros sem resize
fig, axes = plt.subplots(ncols=3, nrows=2)
axes[0][0].imshow(edge_roberts, cmap='gray')
axes[0][0].set_title('Roberts')
axes[0][1].imshow(edge_sobel, cmap='gray')
axes[0][1].set_title('Sobel')
axes[0][2].imshow(edge_sato_false, cmap='gray')
axes[0][2].set_title('Sato - False')
axes[1][0].imshow(edge_scharr, cmap='gray')
axes[1][0].set_title('Scharr')
axes[1][1].imshow(edge_prewitt, cmap='gray')
axes[1][1].set_title('Prewitt')
axes[1][2].imshow(edge_sato_true, cmap='gray')
axes[1][2].set_title('Sato - True')
# Ajusta o plot pra não sobrepor imagens
plt.tight_layout()
plt.show()
# plt.savefig('filtros sem resize')


# Resultado dos filtros com resize
fig, axes = plt.subplots(ncols=3, nrows=2)
axes[0][0].imshow(edge_roberts_res, cmap='gray')
axes[0][0].set_title('Roberts')
axes[0][1].imshow(edge_sobel_res, cmap='gray')
axes[0][1].set_title('Sobel')
axes[0][2].imshow(edge_sato_false_res, cmap='gray')
axes[0][2].set_title('Sato - False')
axes[1][0].imshow(edge_scharr_res, cmap='gray')
axes[1][0].set_title('Scharr')
axes[1][1].imshow(edge_prewitt_res, cmap='gray')
axes[1][1].set_title('Prewitt')
axes[1][2].imshow(edge_sato_true_res, cmap='gray')
axes[1][2].set_title('Sato - True')
# Ajusta o plot pra não sobrepor imagens
plt.tight_layout()
plt.show()
# plt.savefig('filtros com resize')

