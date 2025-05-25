import numpy as np
from skimage.morphology import erosion, dilation, reconstruction
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
from matplotlib import pyplot as plt

# Ouvrir et afficher une image en niveau de gris

# Elements structurant élémentaire en 4-connexité
se1 = diamond(1) 

img =  Image.open('./images/Alan_Turing_photo.jpg')
plt.subplot(131)
plt.imshow(img,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Image originale')

AlanEro = erosion(img, se1)
AlanDil = dilation(img, se1)
AlanGrad = AlanDil - AlanEro
AlanLap = AlanDil.astype(np.int16) + AlanEro.astype(np.int16) - 2*np.asarray(img).astype(np.int16)
plt.subplot(132)
plt.imshow(AlanGrad,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Gradient morphologique')
plt.subplot(133)
plt.imshow(AlanLap,cmap = 'gray',vmin = -50.0,vmax = 50.0)
plt.title('Laplacien morphologique')
plt.show()

Lap_plus = (AlanLap >= 0)
Lap_moins = (AlanLap < 0)
ZeroLap_Alan = dilation(Lap_plus, se1) & Lap_moins
#print(np.unique(ZeroLap_Alan))
plt.subplot(131)
plt.imshow(Lap_plus,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Laplacien >= 0')
plt.subplot(132)
plt.imshow(Lap_moins,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Laplacien < 0')
plt.subplot(133)
plt.imshow(ZeroLap_Alan,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Passages par zéros')
plt.show()

Seuil_bas = 10;
Seuil_haut = 30;
Grad_bas = AlanGrad > Seuil_bas # Lieu de gradient > Seuil_bas
Grad_haut = AlanGrad > Seuil_haut # Lieu de gradient > Seuil_haut
Contours_bas = ZeroLap_Alan & Grad_bas
Contours_haut = ZeroLap_Alan & Grad_haut
# Reconstruction en 8-connexité par défaut
Contours = reconstruction(Contours_haut,Contours_bas)
plt.subplot(131)
plt.imshow(Contours_bas,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Seuil bas')
plt.subplot(132)
plt.imshow(Contours_haut,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Seuil haut')
plt.subplot(133)
plt.imshow(Contours,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Contours')
plt.show()
