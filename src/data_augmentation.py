import os
import PIL
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from skimage.color import rgb2gray
from skimage.util import random_noise
from skimage.transform import rotate
from scipy import ndimage

# Fonction d'augmentation des données avec fonctions flip et mirror
def augment_images_in_path(path):
    for imagefile in os.listdir(path):
        image = Image.open(path+imagefile).convert("RGB")
        if "_flip" in imagefile: 
            continue
        else:
            image2 = image.copy()
            im_flip = ImageOps.flip(image2)
            im_flip.save(path+imagefile+"_flip.jpg")
        if "_mirror" in imagefile: 
            continue
        else:
            image2 = image.copy()
            im_mirror = ImageOps.mirror(image2)
            im_mirror.save(path+imagefile+"_mirror.jpg")
        

augment_images_in_path("Data/Mer/")
augment_images_in_path("Data/Ailleurs/")