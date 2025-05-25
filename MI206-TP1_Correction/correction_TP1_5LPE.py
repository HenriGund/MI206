import numpy as np
from skimage.morphology import erosion, dilation, reconstruction
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.morphology import local_minima, label
from skimage.segmentation import watershed
from skimage import color
from PIL import Image
from scipy import ndimage as ndi
from matplotlib import pyplot as plt

# Ouvrir une image en niveau de gris et conversion en tableau numpy
# au format uint8 (entier non signé entre 0 et 255)
img_ui =  np.asarray(Image.open('./images/uranium.png')).astype(np.uint8)
# Calcul des minima régionaux (en 8-connexité), par formule duale
img_rec = reconstruction(255 - img_ui,255 - img_ui + 1, selem=diamond(1))
ui_min_reg = (img_rec != 255 - img_ui + 1)
seeds = label(ui_min_reg,neighbors = 8) # Etiquetage des minima
ui_ws = watershed(img_ui,seeds) # LPE par défaut : Marqueur = Minima Régionaux
ws_display=color.label2rgb(ui_ws,img_ui)
# Affichage avec matplotlib
plt.subplot(131)
plt.imshow(img_ui,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Originale')
plt.subplot(132)
plt.imshow(ui_min_reg,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Minima régionaux (8-cx)')
plt.subplot(133)
plt.imshow(ws_display)
plt.title('Ligne de Partage des Eaux (brute)')
plt.show()

# Fermeture par reconstruction
se = square(3) # Détermine la taille minimum d'un minimum régional
ui_close = erosion(dilation(img_ui,se),se) # Fermeture morphologique
ui_close_reco = 255 - reconstruction(255 - ui_close,255 - img_ui,selem = diamond(1))
# Calcul des nouveaux minima régionaux
img2_rec = reconstruction(255-ui_close_reco,255-ui_close_reco+1,selem = diamond(1))
ui2_min_reg = (img2_rec != 255-ui_close_reco+1)
seeds2 = label(ui2_min_reg,neighbors = 8) # Etiquetage des minima
ui2_ws = watershed(ui_close_reco,seeds2) # LPE par défaut : Marqueur = Minima Régionaux
ws2_display=color.label2rgb(ui2_ws,img_ui)
# Affichage avec matplotlib
plt.subplot(131)
plt.imshow(ui_close_reco,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Fermeture par reconstruction')
plt.subplot(132)
plt.imshow(ui2_min_reg,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Minima régionaux (8-cx)')
plt.subplot(133)
plt.imshow(ws2_display)
plt.title('Ligne de Partage des Eaux (filtrée 1)')
plt.show()

# Filtrage de Dynamique
prof_min = 60 # Détermine la profondeur minimum d'un bassin versant
ui2_close_reco = reconstruction(ui_close_reco.astype(np.int16),ui_close_reco.astype(np.int16) + prof_min)
# Calcul des nouveaux minima régionaux
img3_rec = reconstruction(255-ui2_close_reco,255-ui2_close_reco+1,selem = diamond(1))
ui3_min_reg = (img3_rec != 255-ui2_close_reco+1)
seeds3 = label(ui3_min_reg,neighbors = 8) # Etiquetage des minima
ui3_ws = watershed(ui2_close_reco,seeds3) # LPE par défaut : Marqueur = Minima Régionaux
ws3_display=color.label2rgb(ui3_ws,img_ui)
# Affichage avec matplotlib
plt.subplot(131)
plt.imshow(ui2_close_reco,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Filtrage de Dynamique')
plt.subplot(132)
plt.imshow(ui3_min_reg,cmap = 'gray',vmin = 0.0,vmax = 1.0)
plt.title('Minima régionaux (8-cx)')
plt.subplot(133)
plt.imshow(ws3_display)
plt.title('Ligne de Partage des Eaux (filtrée 2)')
plt.show()


