#!/usr/bin/env python
# coding: utf-8

# Probabilistic Hough Transform

import numpy as np

from copy import deepcopy
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import color, exposure, io
from skimage.draw import line
from skimage.feature import canny
from skimage.filters.edges import binary_erosion
from skimage.transform import probabilistic_hough_line

# Line finding using the Probabilistic Hough Transform
#image = io.imread('letters-dataset/letters/01_14.png')
#image = io.imread('letters-dataset/letters/01_19.png')
#image = io.imread('letters-dataset/letters/01_24.png') #xadrez, falhou
image = io.imread('letters-dataset/letters/01_35.png') #xadrez, funcionou sem exposure
#image = io.imread('letters-dataset/letters/01_28.png')
image = color.rgb2gray(image)
image = (image*255)
#image = exposure.equalize_hist(image, nbins=2)

edges = canny(image, sigma=0.5)
lines_reference = probabilistic_hough_line(edges, threshold=20, line_length=28, line_gap=8)
drawed_lines = np.zeros_like(image, dtype=np.uint8)
for single_line in lines_reference:
    p0, p1 = single_line
    row, column = line(p0[1], p0[0], p1[1], p1[0])
    drawed_lines[row, column] = 1
image[drawed_lines == 1] = 1

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(drawed_lines, cmap=cm.gray)
ax[2].set_title('Hough Transform')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()



