import numpy as np
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
os.chdir('E:/pictures/bash')
# Open the input image as numpy array, convert to RGB
img=Image.open("a0.png").convert("RGB")
npImage=np.array(img)
h,w=img.size
imgg = mpimg.imread('a0.png')
imgplot = plt.imshow(imgg)

# Create same size alpha layer with circle
alpha = Image.new('L', img.size,0)
draw = ImageDraw.Draw(alpha)
draw.pieslice([145,145,395,395],0,360,fill=255)

# Convert alpha Image to numpy array
npAlpha=np.array(alpha)

# Add alpha layer to RGB
npImage=np.dstack((npImage,npAlpha))

# Save with alpha
Image.fromarray(npImage).save('result.png')