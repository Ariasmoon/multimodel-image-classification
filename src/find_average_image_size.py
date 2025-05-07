import os
from PIL import Image

image_list = []

for imagefile in os.listdir("Data/Mer/"):
    image_list.append("Data/Mer/"+imagefile)
for imagefile in os.listdir("Data/Ailleurs/"):
    image_list.append("Data/Ailleurs/"+imagefile)

sum_width = 0
sum_height = 0
for imagefile in image_list:
    image = Image.open(imagefile).convert('RGB')
    width, height = image.size
    sum_width += width
    sum_height += height

print(round(sum_width/len(image_list)), round(sum_height/len(image_list)))