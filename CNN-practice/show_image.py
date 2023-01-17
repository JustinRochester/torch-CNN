import torchvision.datasets.mnist as mnist
from PIL import Image
import numpy as np


def show_image(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


data = mnist.read_image_file("../data/MNISTS/raw/train-images-idx3-ubyte")
print(data.shape)

img = data[1]

a = np.array(img)
print(type(a))
show_image(a)
