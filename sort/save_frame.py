import cv2
from tqdm import tqdm
import os
import sys

if __name__ == '__main__':
    # Input video
    video = cv2.VideoCapture("prova2.mp4")
    length_input = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    count = 1
    ret = True
    with tqdm(total=length_input, file=sys.stdout) as pbar:
        while ret:
            ret, frame = video.read()

            if not ret:
                continue

            cv2.imwrite('mot_benchmark/train/basket/img1/%06d.jpg'%(count), frame) 

            count+=1