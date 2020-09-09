#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: victor

Abordagem para segmentação da base de dados, utilizando o KMeans.
Tentativa de remoção de linhas/manchas na imagem.

Referencia:
    https://scikit-learn.org/stable/modules/clustering.html#k-means
"""

from skimage import io, transform, color, exposure, filters
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os


## Segmentação por K-Means
def kMeans(img):
    w, h, d = img.shape
    image_array = np.reshape(img, (w * h, d))      # Transformando em "vetor" - necessário
    kmeans = KMeans(n_clusters=2, init='random', n_init=1).fit(image_array)
    labels = kmeans.predict(image_array)
    seg1 = np.reshape(labels, (w,h))
    seg1 = (seg1*255).astype('uint8')
    return seg1
    
## Teste manual
def ajusteManual(im, baixo, alto):
    im2 = im.copy()
    im2[im2<=baixo] = 0
    im2[im2>=alto] = 255

    im2[im2>baixo] = 255*(im2[im2>baixo] - baixo)/(alto - baixo)
    return im2.astype('uint8')


## Filters
def filters_application(image):
    original = image.copy()
    image = color.rgb2gray(image)            # dtype('float64')
    image = (image * 255).astype('uint8')    # Converter para uint8
    
    # Resize da imagem
    image_res= transform.resize(image, (300,300))

    # Filtros com resize
    edge_roberts_res = filters.roberts(image_res)
    edge_sobel_res = filters.sobel(image_res)
    edge_scharr_res = filters.scharr(image_res)
    edge_prewitt_res = filters.prewitt(image_res)
    edge_sato_true_res = filters.sato(image_res, sigmas=range(1,6,1), black_ridges=True)
      
    ##Resultado dos filtros com resize
    fig, axes = plt.subplots(ncols=3, nrows=2)
    axes[0][0].imshow(original)
    axes[0][0].set_title('Original')
    axes[0][1].imshow(edge_roberts_res, cmap='gray')
    axes[0][1].set_title('Roberts')
    axes[0][2].imshow(edge_sobel_res, cmap='gray')
    axes[0][2].set_title('Sobel')
    axes[1][0].imshow(edge_scharr_res, cmap='gray')
    axes[1][0].set_title('Scharr')
    axes[1][1].imshow(edge_prewitt_res, cmap='gray')
    axes[1][1].set_title('Prewitt')
    axes[1][2].imshow(edge_sato_true_res, cmap='gray')
    axes[1][2].set_title('Sato')
    

    
    # plt.subplot(321)
    # plt.imshow(original, cmap='gray')
    # plt.set_title('Original')
    # plt.subplot(322)
    # plt.imshow(edge_roberts_res, cmap='gray')
    # plt.set_title('Roberts')
    # plt.subplot(323)
    # plt.imshow(edge_sobel_res, cmap='gray')
    # plt.set_title('Sobel')
    # plt.subplot(324)
    # plt.set_title('Sobel')
    # plt.imshow(edge_scharr_res, cmap='gray')
    # plt.subplot(325)
    # plt.imshow(edge_prewitt_res, cmap='gray')
    # plt.set_title('Prewitt')
    # plt.subplot(326)
    # plt.imshow(edge_sato_true_res, cmap='gray')
    # plt.set_title('Sato')

    # Ajusta o plot pra não sobrepor imagens
    plt.tight_layout()
    plt.savefig(maindir +'filter_results/'+ dire)
    plt.close()





# # ## Para testar apenas uma imagem:
# path = '/media/victor/1f3aa121-0188-42a3-b7c4-b271c3c0afca/2020 - 1/Processamento de imagens/repositorio_github/letters/example_dataset/01_256.png'
# img = io.imread(path)

# # Resize - Ajuda a melhorar a resposta ao mudar o tamanho de 32x32 => 300x300
# img = transform.resize(img, (300,300))

# img2 = kMeans(img)
# plt.figure()
# plt.imshow(img, cmap='gray')

# plt.subplot(121)     
# plt.imshow(img)
# plt.title("Imagem original")
# plt.subplot(122)
# plt.imshow(img2, cmap="gray")
# plt.title("Resultado")


## Teste automatizado:
maindir = '/media/victor/1f3aa121-0188-42a3-b7c4-b271c3c0afca/2020 - 1/Processamento de imagens/repositorio_github/letters/example_dataset/'
files = os.listdir(maindir)

## Escolha do método a ser utilizado
choose = 'filters'
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
        elif (choose == 'filters'):
            im = io.imread(maindir + dire)
            filters_application(im)              