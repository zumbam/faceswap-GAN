# this file has the use to avoid jupiter notebooks session problems

import keras.backend as K
from detector.face_detector import MTCNNFaceDetector
import glob
import os
from preprocess import preprocess_video

TOTAL_ITERS = 34000




sess = K.get_session()

fd = MTCNNFaceDetector(sess=sess, model_path="./mtcnn_weights/")


fn_source_video = 'D:\QSync\Master_Studium\Semester3\Hauptseminar\Data\Trump\Rede.mp4'
res = os.path.exists(fn_source_video)
print(res)
fn_target_video = 'D:\QSync\Master_Studium\Semester3\Hauptseminar\Data\Video_me\test_videp.mp4'
res = os.path.exists(fn_source_video)
print(res)

os.mkdirs("faceA/rgb")
os.mkdirs("faceA/binary_mask")
os.mkdirs("faceB/rgb")
os.mkdirs("faceB/binary_mask")


save_interval = 5 # perform face detection every {save_interval} frames
save_path = "./faceA/"
preprocess_video(fn_source_video, fd, save_interval, save_path)
save_path = "./faceB/"
preprocess_video(fn_target_video, fd, save_interval, save_path)