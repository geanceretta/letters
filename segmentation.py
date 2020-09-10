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
from skimage.segmentation import active_contour
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import morphology
from skimage.morphology import dilation, diamond, square, erosion, disk
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
    im2= transform.resize(im2, (300,300))
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
    # Ajusta o plot pra não sobrepor imagens
    plt.tight_layout()
    plt.savefig(maindir +'filter_results/'+ dire)
    plt.close()


def snake(img):
    img = transform.resize(img, (300,300))
    img = color.rgb2gray(img)
    s = np.linspace(0, 2*np.pi, 400)
    r = 150 + 120*np.sin(s)
    c = 150 + 120*np.cos(s)
    init = np.array([r, c]).T
    
    snake = active_contour(img,init, alpha=0.015, beta=10, gamma=0.001)
    
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])   

    plt.savefig(maindir +'snake_results/'+ dire)
    plt.close()


def hyst_threshold(image):
    original = image.copy()
    image = color.rgb2gray(image)                        # dtype('float64')
    image = (image * 255).astype('uint8')                # Converter para uint8
    # image_res = transform.resize(image, (300,300))
    image_res = (image * 255).astype('uint8')            
    
   
    ## Filtro Prewitt
    edges_p = filters.prewitt(image_res)
    
    # Parametros - Removeu linha
    ## Imagem tamanho original
    # low_p = 0.1
    # high_p = 0.8
    ## Imagem com resize
    low_p = 0.014
    high_p = 0.1
    
    lowt_p = (edges_p > low_p).astype(int)
    hight_p = (edges_p > high_p).astype(int)
    hyst_p = filters.apply_hysteresis_threshold(edges_p, low_p, high_p)
    
    ## Filtro Sobel
    edges_s = filters.sobel(image_res)
    
    low_s = low_p
    high_s = high_p
    
    lowt_s = (edges_s > low_s).astype(int)
    hight_s = (edges_s > high_s).astype(int)
    hyst_s = filters.apply_hysteresis_threshold(edges_s, low_s, high_s)
    
    ### Fechamento
    pretty_d = lowt_p.copy()
    sobel_d = lowt_s.copy()

    # Dilatação
    pretty_d = dilation(pretty_d, diamond(8))
    sobel_d = dilation(sobel_d, diamond(8))

    # Erosão sobre a dilatação = Fechamento
    closed_pretty = erosion(pretty_d, diamond(8))
    closed_sobel = erosion(sobel_d, diamond(8))
    
    # Plot
    fig, ax = plt.subplots(nrows=2, ncols=3)
    ax[0][0].imshow(original)
    ax[0][0].set_title('Imagem original')    
    ax[0][1].imshow(lowt_p, cmap='gray')
    ax[0][1].set_title('Prewitt')   
    ax[0][2].imshow(lowt_s, cmap='gray')
    ax[0][2].set_title('Sobel')
    ax[1][0].imshow(original)
    # ax[1][0].set_title('Imagem original')    
    ax[1][1].imshow(closed_pretty, cmap='gray')
    # ax[1][1].set_title('Prewitt')   
    ax[1][2].imshow(closed_sobel, cmap='gray')
    # ax[2][1].set_title('Sobel')
    
    plt.tight_layout()
    plt.savefig(maindir +'hyst_threshold_results/'+ dire)
    plt.close()
    




## Teste automatizado:
maindir = '/media/victor/1f3aa121-0188-42a3-b7c4-b271c3c0afca/2020 - 1/Processamento de imagens/repositorio_github/letters/example_dataset/'
files = os.listdir(maindir)

## Escolha do método a ser utilizado
choose = 'hyst_threshold'
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
            plt.close()
        elif (choose == 'ajusteManual'):
            im = io.imread(maindir + dire)
            im2 = transform.resize(im, (300,300)) 
            im3 = color.rgb2gray(im2)
            im3 = (im3*255).astype('uint8')
            
            # Equalize Hist
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
            plt.close()
        elif (choose == 'filters'):
            im = io.imread(maindir + dire)
            filters_application(im)     
        elif (choose == 'snake'):
            im = io.imread(maindir + dire)
            snake(im)
        elif (choose == 'hyst_threshold'):
            im = io.imread(maindir + dire)
            im2 = transform.resize(im, (300,300))
            hyst_threshold(im2)               
            
            
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