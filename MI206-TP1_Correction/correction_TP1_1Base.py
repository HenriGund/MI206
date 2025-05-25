import numpy as np
from skimage.morphology import erosion, dilation
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from PIL import Image
from matplotlib import pyplot as plt

from skimage.util import img_as_ubyte

# Definition de différents elements structurants plats
se1 = square(5) # square

se2 = diamond(4) # diamond

se3  = np.ones((7)) # line

se4 = disk(8)

se5 = np.array([[0, 0, 1, 1, 1],
               [0, 1, 0, 0, 0,],
               [0, 1, 1, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 0, 1, 1, 1]], dtype=np.uint8)

#Ouvrir et afficher une image en niveau de gris
img =  np.asarray(Image.open('./images/clock.png')).astype(np.uint8)
plt.subplot(241)
plt.imshow(img,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Image originale')

imDil = dilation(img,se1) # Dilatation morphologique
imEro = erosion(img, se1)

plt.subplot(242)
plt.imshow(imDil,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Dilatation')
plt.subplot(243)
plt.imshow(imEro,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Erosion')

imgradient=imDil-imEro
plt.subplot(244)
plt.imshow(imgradient,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Gradient Morphologique')
A=2*np.asarray(img).astype(np.int16)
laplacien=imDil.astype(np.int16)+imEro.astype(np.int16)-A
plt.subplot(245)
plt.imshow(laplacien,cmap = 'gray',vmin = -255.0,vmax = 255.0)
plt.title('Laplacien Morphologique')

imclose = Image.fromarray(erosion(dilation(img,se1))) # Ouverture morphologique
imopen = Image.fromarray(dilation(erosion(img, se1)))

plt.subplot(246)
plt.imshow(imclose,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Fermeture')
plt.subplot(247)
plt.imshow(imopen,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Ouverture')

# DUALITY CHECKING
dual_img= (255-np.asarray(img).astype(np.float32)).astype(np.uint8)
imEro_2 = erosion(dual_img, se1)
dual_ero_2=(255-imEro_2.astype(np.float32)).astype(np.uint8)
imout=imDil-dual_ero_2
plt.subplot(248)
plt.imshow(imout,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Test de dualité')
print("La somme de l'image de différence vaut :",imout.sum())
plt.show()

# You can do the same for the checking the opening / closing!
