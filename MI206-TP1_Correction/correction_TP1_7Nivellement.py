import numpy as np
from skimage.morphology import erosion, dilation, reconstruction
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from PIL import Image
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
from matplotlib import pyplot as plt


#Ouvrir une image en niveau de gris et convertir en tableau numpy
img =  np.asarray(Image.open('./images/goldhill.png')).astype(np.uint8)
imFAS=img.copy()
taille_max = 8 # Calcul du filtre alterné séquentiel (Ouverture d'abord)
for i in range(1,taille_max):
  se = disk(i);
  imFAS = erosion(imFAS,se);
  imFAS = dilation(imFAS,se);
  imFAS = dilation(imFAS,se);
  imFAS = erosion(imFAS,se);

imFASmask = (imFAS > img) 
imFASsup = imFASmask*imFAS + ~imFASmask*255 # Image partout supérieure (fond à 255)
imFASinf = ~imFASmask*imFAS # Image partout inférieure (fond à 0)

imFASinf_reco = reconstruction(imFASinf,img) # Reconstruction directe
imFASsup_reco = 255 - reconstruction(255 - imFASsup,255 - img) # Reconstruction duale

imNivel = imFASmask*imFASsup_reco + ~imFASmask*imFASinf_reco  # Recombinaison

plt.subplot(131)
plt.imshow(img,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Image originale')
plt.subplot(132)
plt.imshow(imFAS,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Filtre Alterné Séquentiel')
plt.subplot(133)
plt.imshow(imNivel,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Nivellement')
plt.show()

