from __future__ import print_function
import cv2
import numpy as np
import time

import sys
sys.path.insert(0, "/home/weak_labeler/")
sys.path.insert(0, "/home/video_toolkit/")

from WeakLabeler.tracker_backend import MultiObjectTracker

from VideoToolkit.IO import H264LossLessReader, H264LossLessSaver
from VideoToolkit.tools import frames_to_multiscene, get_cv_resize_function, \
    rescal_to_image
from VideoToolkit.bbox_ops import expand_bbox



def display(image):
    cv2.imshow('bg1', image)
    keyboard = cv2.waitKey(1000)
    if keyboard == 'q' or keyboard == 27:
        breakeyboard = cv2.waav
        

if __name__ == "__main__":
    filename = "/etc/ext/mnt/ext/Drive\ D/work/IPM/data/5002-02\ large.m4v"
    reader = H264LossLessReader(input_filename=filename,
                                width_height=None,
                                fps=None)

    writer = None

    resizer_func = get_cv_resize_function()
    res_frame_shape = (600, 1024) # height, width
    # writer = H264LossLessSaver("stat_based_bg1_3000_01",
    #                             (reader.width, reader.height),
    #                             20,
    #                             compression_rate=25)

    motracker = None

    count = 0


    while True:
        frame_c = reader.read()
        if frame_c is None:
            print("*******Got None frame")
            break

        if motracker is None:
            motracker = MultiObjectTracker(frame_c.shape[:-1])

        motracker.update(frame_c)
        print(f"****currently tracking objects #{len(motracker.objects)}****")

        mask = np.where(motracker.get_mask() == 1, 255, 0).astype('uint8')

        result = frame_c.copy()
        for obj in motracker.objects:
            box = obj.bbox
            cv2.rectangle(result, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 2)
            cv2.putText(result, #numpy array on which text is written
                            f"{obj.tracker_ID}", #text
                            (box[0], box[1]-5), #position at which writing has to start
                            cv2.FONT_HERSHEY_SIMPLEX, #font family
                            .6, #font size
                            (10, 10, 255, 255), #font color
                            2) # font strok

        patched_res = frame_c.copy()
        patches = []
        for obj in motracker.objects:
            box = obj.bbox
            box = expand_bbox(box[0], box[1], box[2], box[3],
                            (patched_res.shape[1], patched_res.shape[0]),
                            to_size=(256, 256))
            patches.append(box)
            cv2.rectangle(patched_res, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 2)
            cv2.putText(patched_res, #numpy array on which text is written
                            f"{obj.tracker_ID}", #text
                            (box[0], box[1]-5), #position at which writing has to start
                            cv2.FONT_HERSHEY_SIMPLEX, #font family
                            .6, #font size
                            (10, 10, 255, 255), #font color
                            2) # font strok

        frame_list = [cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB),
                        result,
                        patched_res]
        texts = [f"mask({count})",
                "detected boxes",
                "patches"]
        res_frame, res_shape = frames_to_multiscene(frame_list,
                                                    texts=texts,
                                                    resulting_frame_shape=res_frame_shape,
                                                    method='horizontal',
                                                    grid_dim=(1, 3),
                                                    resizer_func=resizer_func)
    

        # if writer is None:
        #     writer = H264LossLessSaver("multiframe",
        #                                 (res_shape[1], res_shape[0]),
        #                                 20,
        #                                 compression_rate=25)
        # writer.write(res_frame)
        
        
        # show thresh and result    
        # cv2.imshow('bg1', res_frame)
        # keyboard = cv2.waitKey(30)
        # if keyboard == 'q' or keyboard == 27:
        #     breakeyboard = cv2.waav
        display(res_frame)


        # save patches
        for i, box in enumerate(patches):
            res = frame_c[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
            cv2.imwrite(f"res/{count}_{i}.png", res, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # if count > 100:
        #     break
        count +=1