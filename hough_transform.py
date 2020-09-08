#!/usr/bin/env python
# coding: utf-8

# # Hough Transform

# ## Probabilistic

from matplotlib import cm
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import color, exposure, io
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

# Line finding using the Probabilistic Hough Transform
image = io.imread('letters-dataset/letters/01_14.png')
image = color.rgb2gray(image)
image = (image*255)
image = exposure.equalize_hist(image, nbins=2)
edges = canny(image, sigma=0.5)
lines = probabilistic_hough_line(edges, threshold=20, line_length=28, line_gap=8)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()


# ### Dúvida: Como remover as linhas da imagem original baseado no resultado anterior?

# Tentei obter um Numpy Array baseado no plot do matplotlib, porém sem sucesso.
# Referência: https://stackoverflow.com/questions/20130768/retrieve-xy-data-from-matplotlib-figure

# In[119]:


#lines_mask = np.zeros_like(lines, dtype='bool')
#plot_image = plt.imshow(edges * 0)
#for line in lines:
#    p0, p1 = line
#    lines_plot = plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
#
#lines_array = lines_plot[0].get_data()
#lines_array
#
#plt.figure()
#plt.imshow(lines_array * 255, cmap='gray', vmin=0, vmax=255)

#lines_image = plt.get_xydata()
#lines_plot.set_xlim((0, image.shape[1]))
#lines_plot.set_ylim((image.shape[0], 0))



