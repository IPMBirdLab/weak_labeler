from __future__ import print_function
import cv2
import numpy as np
import time

import sys
sys.path.insert(0, "/home/devuser/Documents/workspace/weak_labeler/")
sys.path.insert(0, "/home/devuser/Documents/workspace/video_toolkit/")

from WeakLabeler.matcher_backend import Matcher

from VideoToolkit.IO import H264LossLessReader, H264LossLessSaver
from VideoToolkit.tools import frames_to_multiscene, get_cv_resize_function, \
    rescal_to_image
from VideoToolkit.bbox_ops import expand_bbox, convert_bbox

import random


def generate_random_bboxes(img_shape, num_boxes, box_size=(265, 265)):
    """[summary]

    Parameters
    ----------
    img_shape : [type]
        numpy compatible shape (height, width)
    num_boxes : int
        number of boxes to generate
    box_size : tuple, optional
        default bounding box size. a numpy compatible shape (height, width),
        by default (265, 265)
        Note: only constant box size is suported for now
    """    
    bboxes = []
    for i in range(num_boxes):
        y = random.randint(0, img_shape[0]-box_size[0])
        x = random.randint(0, img_shape[1]-box_size[1])
        bbox = (x, y, box_size[1], box_size[0])
        bbox = convert_bbox(bbox, in_fmt='xywh', out_fmt='xyxy')
        bboxes.append(bbox)
    
    return bboxes



def display(image):
    cv2.imshow('bg1', image)
    keyboard = cv2.waitKey(1000)
    if keyboard == 'q' or keyboard == 27:
        breakeyboard = cv2.waav
        

if __name__ == "__main__":
    filename = "/home/devuser/Documents/workspace/data/experiment8/experiment8_1600_1200.m4v"
    reader = H264LossLessReader(input_filename=filename,
                                width_height=None,
                                fps=None)

    writer = None
    matcher = Matcher(max_frame_distance=100) 

    resizer_func = get_cv_resize_function()
    res_frame_shape = (600, 1024) # height, width

    count = 0
    while True:
        frame_c = reader.read()
        if frame_c is None:
            print("*******Got None frame")
            break

        patches = generate_random_bboxes(frame_c.shape[:-1],
                                        int(frame_c.shape[0]/256),
                                        box_size=(256, 256))

        rnd_patches = frame_c.copy()
        for box in patches:
            cv2.rectangle(rnd_patches, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

        unique_bboxes = matcher.get_unique_patches(frame_c, patches)
        print(f"****current unique objects #{len(unique_bboxes)}****")
        unique_patched_res = frame_c.copy()
        for box in unique_bboxes:
            cv2.rectangle(unique_patched_res, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)



        print(f"****current frames in history #{len(matcher.unique_patches)}****")
        bbox_history = []
        for unq in matcher.unique_patches:
            for box in unq.bboxes:
                bbox_history.append(box)

        history_patches = frame_c.copy()
        for box in bbox_history:
            cv2.rectangle(history_patches, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)


        frame_list = [unique_patched_res,
                      rnd_patches,
                      history_patches]
        texts = [f"unique({count})",
                "random",
                "history"]
        res_frame, res_shape = frames_to_multiscene(frame_list,
                                                    texts=texts,
                                                    resulting_frame_shape=res_frame_shape,
                                                    method='horizontal',
                                                    grid_dim=(2, 3),
                                                    resizer_func=resizer_func)
    

        if writer is None:
            writer = H264LossLessSaver("exp8_neg",
                                        (res_shape[1], res_shape[0]),
                                        25,
                                        compression_rate=25)
        writer.write(res_frame)
        
        display(res_frame)

        for i, box in enumerate(unique_bboxes):
            res = frame_c[box[1]:box[3], box[0]:box[2]]
            cv2.imwrite(f"/home/devuser/Documents/workspace/data/experiment8/res/{count}_{i}.png", res, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # if count > 100:
        #     break
        count +=1
