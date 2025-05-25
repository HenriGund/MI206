import numpy as np
from skimage.morphology import erosion, dilation
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from PIL import Image
from skimage.util import img_as_ubyte
from matplotlib import pyplot as plt

# Elements structurant déterminant la portée spatiale du contraste
se1 = disk(3) 

Alan =  np.asarray(Image.open('./images/Alan_Turing_photo.jpg')).astype(np.uint8)
AlanEro = erosion(Alan, se1)
AlanDil = dilation(Alan, se1)
AlanMask = ((AlanDil - Alan) < (Alan - AlanEro)) 
AlanContraste = AlanMask*AlanDil + ~AlanMask*AlanEro

plt.subplot(141)
plt.imshow(Alan,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Image originale')
plt.subplot(142)
plt.imshow(AlanEro,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Erosion')
plt.subplot(143)
plt.imshow(AlanDil,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Dilatation')
plt.subplot(144)
plt.imshow(AlanContraste,cmap = 'gray',vmin = 0.0,vmax = 255.0)
plt.title('Réhaussement de contraste')
plt.show()
