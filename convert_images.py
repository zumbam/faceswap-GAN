# this file has the use to avoid jupiter notebooks session problems

import keras.backend as K
from detector.face_detector import MTCNNFaceDetector
import glob
import os
from preprocess import preprocess_video

os.environ["CUDA_VISIBLE_DEVICES"]="1"

TOTAL_ITERS = 34000

ML_Server_Trump_Video_Path = '/home/ronczka/Data/Trump/Rede.mp4'
ML_Server_Me_Video_Path = '/home/ronczka/Data/Me/test_videp.mp4'

Local_Trump_Video_Path = 'D:\QSync\Master_Studium\Semester3\Hauptseminar\Data\Trump\Rede.mp4'
Local_Me_Video_Path = 'D:\QSync\Master_Studium\Semester3\Hauptseminar\Data\Video_me\test_videp.mp4'


sess = K.get_session()

fd = MTCNNFaceDetector(sess=sess, model_path="./mtcnn_weights/")


fn_source_video = ML_Server_Me_Video_Path
res = os.path.exists(fn_source_video)
print(res)
fn_target_video = ML_Server_Trump_Video_Path
res = os.path.exists(fn_source_video)
print(res)

os.makedirs("faceA/rgb")
os.makedirs("faceA/binary_mask")
os.makedirs("faceB/rgb")
os.makedirs("faceB/binary_mask")


save_interval = 5 # perform face detection every {save_interval} frames
save_path = "./faceA/"
preprocess_video(fn_source_video, fd, save_interval, save_path)
save_path = "./faceB/"
preprocess_video(fn_target_video, fd, save_interval, save_path)
