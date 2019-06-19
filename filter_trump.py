import os
from PIL import Image
import numpy as np
from shutil import copy2

if __name__=='__main__':
    trump_image_path = 'trump.jpg'
    trump_related_image_path = 'trump_related.jpg'
    trump_image = Image.open(trump_image_path)
    trump_image = np.array(trump_image.getdata())
    trump_related_image = Image.open(trump_related_image_path)
    trump_related_image= np.array(trump_related_image.getdata())

    results_path = 'C:/Users/stefa/Videos/GAN_deep_fakes/results'
    images_path = 'C:/Users/stefa/Videos/GAN_deep_fakes/trump_data'
    image_paths = os.listdir(images_path)
    for i in range(len(image_paths)):
        image_path = os.path.join(images_path, image_paths[i])
        image = Image.open(image_path)
        image_array = np.array(image.getdata())
        diff_trump = sum(sum(abs(image_array - trump_image)))
        diff_related = sum(sum(abs(image_array - trump_related_image)))
        if diff_trump < diff_related:
            copy2(image_path, results_path)
        image.close()


    print(image_paths)
