import face_alignment
import cv2
import numpy as np
from glob import glob
from pathlib import PurePath, Path
from matplotlib import pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

dir_faceA = "./faceA"
dir_faceB = "./filtered_faces"
dir_bm_faceA_eyes = "./binary_masks/faceA_eyes"
dir_bm_faceB_eyes = "./binary_masks/faceB_eyes"

fns_faceA = glob(f"{dir_faceA}/*.*")
fns_faceB = glob(f"{dir_faceB}/*.*")

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda', flip_input=False)

os.makedirs(dir_bm_faceA_eyes, exist_ok=True)
os.makedirs(dir_bm_faceB_eyes, exist_ok=True)


fns_face_not_detected = []

for idx, fns in enumerate([fns_faceA, fns_faceB]):
    if idx == 0:
        save_path = dir_bm_faceA_eyes
    elif idx == 1:
        save_path = dir_bm_faceB_eyes

        # create binary mask for each training image
    for fn in fns:
        raw_fn = PurePath(fn).parts[-1]

        x = plt.imread(fn)
        x = cv2.resize(x, (256, 256))
        preds = fa.get_landmarks(x)

        if preds is not None:
            preds = preds[0]
            mask = np.zeros_like(x)

            # Draw right eye binary mask
            pnts_right = [(preds[i, 0], preds[i, 1]) for i in range(36, 42)]
            hull = cv2.convexHull(np.array(pnts_right)).astype(np.int32)
            mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

            # Draw left eye binary mask
            pnts_left = [(preds[i, 0], preds[i, 1]) for i in range(42, 48)]
            hull = cv2.convexHull(np.array(pnts_left)).astype(np.int32)
            mask = cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)

            # Draw mouth binary mask
            # pnts_mouth = [(preds[i,0],preds[i,1]) for i in range(48,60)]
            # hull = cv2.convexHull(np.array(pnts_mouth)).astype(np.int32)
            # mask = cv2.drawContours(mask,[hull],0,(255,255,255),-1)

            mask = cv2.dilate(mask, np.ones((13, 13), np.uint8), iterations=1)
            mask = cv2.GaussianBlur(mask, (7, 7), 0)

        else:
            mask = np.zeros_like(x)
            print(f"No faces were detected in image '{fn}''")
            fns_face_not_detected.append(fn)

        plt.imsave(fname=f"{save_path}/{raw_fn}", arr=mask, format="jpg")

num_faceA = len(glob(dir_faceA+"/*.*"))
num_faceB = len(glob(dir_faceB+"/*.*"))

print("Nuber of processed images: "+ str(num_faceA + num_faceB))
print("Number of image(s) with no face detected: " + str(len(fns_face_not_detected)))
