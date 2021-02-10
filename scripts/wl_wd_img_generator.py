from __future__ import print_function
import cv2
import numpy as np
import math
import time

import sys
sys.path.insert(0, "/home/devuser/Documents/workspace/weak_labeler/")
sys.path.insert(0, "/home/devuser/Documents/workspace/video_toolkit/")

from WeakLabeler.tracker_backend import MultiObjectTracker
from WeakLabeler.matcher_backend import Matcher
from WeakLabeler.stat_bs import FrameWindow

from VideoToolkit.IO import H264LossLessReader, H264LossLessSaver
from VideoToolkit.tools import frames_to_multiscene, get_cv_resize_function, \
    rescal_to_image
from VideoToolkit.bbox_ops import expand_bbox, convert_bbox



def display(image):
    cv2.imshow('bg1', image)
    keyboard = cv2.waitKey(1000)
    if keyboard == 'q' or keyboard == 27:
        breakeyboard = cv2.waav
        

class FrameGenerator(FrameWindow):
    def __init__(self, shape, window_size=100):
        super().__init__(shape, window_size)
        self.current_idx = None

    def set_new_frame(self, frame):
        self.last_idx = self.index
        super().set_new_frame(frame)

    def get_latest_frame(self):
        return self.window[self.last_idx]

    def get_current_frame(self):
        if self.num < self.window_size:
            return None
        if self.current_idx is None:
            self.current_idx = math.floor(self.num / 2)

        self.current_idx = self.fix_index(self.current_idx + 1)
        return self.window[self.current_idx].astype('uint8')



if __name__ == "__main__":
    filename = "/home/devuser/Documents/workspace/data/experiment5/experiment5_1600_1200.m4v"
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

    frame_gen = None

    motracker = None
    matcher = Matcher(max_frame_distance=250)        

    count = 0


    while True:
        frame_n = reader.read()
        if frame_n is None:
            print("*******Got None frame")
            break

        if frame_gen is None:
            frame_gen = FrameGenerator(frame_n.shape, 20)

        frame_gen.set_new_frame(frame_n)

        if motracker is None:
            motracker = MultiObjectTracker(frame_n.shape[:-1])

        motracker.set_latest_window_frame(frame_gen.get_latest_frame())

        frame_c = frame_gen.get_current_frame()
        if frame_c is None:
            continue

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
            box = expand_bbox(convert_bbox(box, in_fmt='xywh', out_fmt='xyxy'),
                            (patched_res.shape[1], patched_res.shape[0]),
                            to_size=(256, 256))
            patches.append(box)
            cv2.rectangle(patched_res, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
            cv2.putText(patched_res, #numpy array on which text is written
                            f"{obj.tracker_ID}", #text
                            (box[0], box[1]-5), #position at which writing has to start
                            cv2.FONT_HERSHEY_SIMPLEX, #font family
                            .6, #font size
                            (10, 10, 255, 255), #font color
                            2) # font strok

        unique_bboxes = matcher.get_unique_patches(frame_c, patches)
        print(f"****current unique objects #{len(unique_bboxes)}****")
        unique_patched_res = frame_c.copy()
        for box in unique_bboxes:
            cv2.rectangle(unique_patched_res, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)


        bg = motracker.statbased.get_bmask(cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY))
        mean = motracker.statbased.long_mean
        var = motracker.statbased.long_var

        fore, back, uncer = motracker.statbased.get_uncertainty_maps(cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY))


        frame_list = [cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB),
                        patched_res,
                        unique_patched_res,
                        cv2.cvtColor(bg, cv2.COLOR_GRAY2RGB),
                        cv2.cvtColor(mean, cv2.COLOR_GRAY2RGB),
                        cv2.cvtColor(var, cv2.COLOR_GRAY2RGB),
                        cv2.cvtColor(rescal_to_image(fore), cv2.COLOR_GRAY2RGB),
                        cv2.cvtColor(rescal_to_image(back), cv2.COLOR_GRAY2RGB),
                        cv2.cvtColor(rescal_to_image(uncer), cv2.COLOR_GRAY2RGB)]
        texts = [f"mask({count})",
                "patches",
                "unique patches",
                "background",
                "mean",
                "variance",
                "foreground",
                "background",
                "uncertain"]
        res_frame, res_shape = frames_to_multiscene(frame_list,
                                                    texts=texts,
                                                    resulting_frame_shape=res_frame_shape,
                                                    method='grid',
                                                    grid_dim=(3, 3),
                                                    resizer_func=resizer_func)
    

        # if writer is None:
        #     writer = H264LossLessSaver("multiframe",
        #                                 (res_shape[1], res_shape[0]),
        #                                 25,
        #                                 compression_rate=25)
        # writer.write(res_frame)
        
        
        # show thresh and result    
        # cv2.imshow('bg1', res_frame)
        # keyboard = cv2.waitKey(30)
        # if keyboard == 'q' or keyboard == 27:
        #     breakeyboard = cv2.waav
        display(res_frame)


        # save patches
        # for i, box in enumerate(patches):
        #     res = frame_c[box[1]:box[1]+box[3], box[0]:box[0]+box[2]]
        #     cv2.imwrite(f"res/{count}_{i}.png", res, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        for i, box in enumerate(unique_bboxes):
            res = frame_c[box[1]:box[3], box[0]:box[2]]
            cv2.imwrite(f"/home/devuser/Documents/workspace/data/experiment5/res/{count}_{i}.png", res, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        # if count > 100:
        #     break
        count +=1