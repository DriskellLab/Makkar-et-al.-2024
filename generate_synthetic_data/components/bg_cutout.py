#!/usr/bin/env python
# coding: utf-8

# In[1]:
from PIL import Image
import random
import os

Image.MAX_IMAGE_PIXELS = None


# In[2]:
im = Image.open("./background.png")


if not os.path.exists('./bg/'):
    os.makedirs(os.path.join('./bg/'))

# In[3]:
width, height = im.size

# In[5]:
for i in range(100):
    x = random.randint(0, width - 1024)
    y = random.randint(0, height - 1024)
    cutout = im.crop((x, y, x + 1024, y + 1024))
    cutout.save("./bg/cutout"+str(i)+".png")
    print("Generated background image number {}".format(str(i)))
