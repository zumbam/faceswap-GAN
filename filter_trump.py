import os
from PIL import Image
import numpy as np
from shutil import copy2

if __name__=='__main__':

    max_epsilon= 0.07
    trump_image_path = 'trump.jpg'

    trump_image = Image.open(trump_image_path)
    

    results_path = 'filtered_faces'
    images_path = './faceB/rgb'

    os.makedirs(results_path, exist_ok=True)
    image_paths = os.listdir(images_path)

    trump_hist = np.array(trump_image.histogram()) / (trump_image.width * trump_image.height * 3)
    for i in range(len(image_paths)):
        image_path = os.path.join(images_path, image_paths[i])
        image = Image.open(image_path)
        image_size = image.size
        #print(image_size)
        image_array = np.array(image.getdata())
        #trump_image_resized = trump_image.resize(image_size)
        #trump_image_resized = np.array(trump_image_resized.getdata())
        image_hist = np.array(image.histogram()) / (image.width * image.height * 3)
        diff_trump = np.linalg.norm(trump_hist - image_hist, ord=2, axis=0)
        print(diff_trump)
        # use a the color histogram for scanning true images containing a trump like colors
        if diff_trump < max_epsilon:
            copy2(image_path, results_path)
        image.close()


    print(image_paths)
