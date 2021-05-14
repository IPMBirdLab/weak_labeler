from __future__ import print_function
import cv2
import numpy as np
import math
import time

import sys

sys.path.insert(0, "/home/devuser/Documents/workspace/weak_labeler/")
sys.path.insert(0, "/home/devuser/Documents/workspace/video_toolkit/")

from WeakLabeler.stat_bs import StatBS
from WeakLabeler.tracker_backend import MultiObjectTracker
from WeakLabeler.matcher_backend import Matcher

from VideoToolkit.IO import H264LossLessReader, H264LossLessSaver
from VideoToolkit.tools import (
    frames_to_multiscene,
    get_cv_resize_function,
    rescal_to_image,
)
from VideoToolkit.bbox_ops import expand_bbox, convert_bbox


def display(image):
    cv2.imshow("bg1", image)
    keyboard = cv2.waitKey(1000)
    if keyboard == "q" or keyboard == 27:
        breakeyboard = cv2.waav


file_num = 30
if __name__ == "__main__":
    filename = f"/workspace8/data/new/experiment{file_num}_1600_1200.m4v"
    reader = H264LossLessReader(input_filename=filename, width_height=None, fps=None)

    writer = None

    resizer_func = get_cv_resize_function()
    res_frame_shape = (600, 1024)  # height, width
    # writer = H264LossLessSaver("stat_based_bg1_3000_01",
    #                             (reader.width, reader.height),
    #                             20,
    #                             compression_rate=25)

    stats_bs = None
    motracker = None
    matcher = Matcher(max_frame_distance=20)

    count = 0
    while True:
        frame_c = reader.read()
        if frame_c is None:
            print("*******Got None frame")
            break

        if stats_bs is None:
            stats_bs = StatBS(frame_c.shape[:-1])

        if motracker is None:
            motracker = MultiObjectTracker(frame_c.shape[:-1])

        frame = cv2.cvtColor(frame_c, cv2.COLOR_BGR2GRAY)

        stats_bs.apply(frame)
        stats_bs.get_bmask(frame)

        mask = stats_bs.get_bmask(frame)
        mean = stats_bs.long_mean
        var = stats_bs.long_var

        fore, back, uncer = stats_bs.get_uncertainty_maps(frame)

        print(f"\n********** mean {np.mean(fore)} **********")
        print(f"********** max {np.max(fore)} **********")
        print(f"********** var {np.var(fore)} **********")

        motracker.set_latest_window_frame(frame_c)
        masks, bboxes = motracker.get_new_bboxs(frame_c)
        print(f"****current blobs #{len(masks)}****")

        temp_boxes = []
        for box in bboxes:
            box = expand_bbox(
                convert_bbox(box, in_fmt="xywh", out_fmt="xyxy"),
                (frame_c.shape[1], frame_c.shape[0]),
                to_size=(256, 256),
            )
            temp_boxes.append(box)

        unique_bboxes = matcher.get_unique_patches(frame_c, temp_boxes)

        unique_patched_res = frame_c.copy()
        for box in unique_bboxes:
            cv2.rectangle(
                unique_patched_res, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2
            )
        print(f"****current unique blobs #{len(unique_bboxes)}****")

        frame_list = [
            cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB),
            unique_patched_res,
            cv2.cvtColor(var, cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(rescal_to_image(fore), cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(rescal_to_image(back), cv2.COLOR_GRAY2RGB),
            cv2.cvtColor(rescal_to_image(uncer), cv2.COLOR_GRAY2RGB),
        ]
        texts = [
            f"mask({count})",
            "unique patched",
            "variance",
            "foreground",
            "background",
            "uncertain",
        ]
        res_frame, res_shape = frames_to_multiscene(
            frame_list,
            texts=texts,
            resulting_frame_shape=res_frame_shape,
            method="grid",
            grid_dim=(2, 3),
            resizer_func=resizer_func,
        )

        if writer is None:
            writer = H264LossLessSaver(
                f"/workspace8/data/weak_label_videos/multiframe{file_num}",
                (res_shape[1], res_shape[0]),
                10,
                compression_rate=25,
            )
        writer.write(res_frame)

        # display(res_frame)

        if len(unique_bboxes) > 0:
            cv2.imwrite(
                f"/workspace8/data/pre_processed/experiment15_t/frames/{count}.png",
                frame_c,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            )
            cv2.imwrite(
                f"/workspace8/data/pre_processed/experiment{file_num}/objness/{count}.png",
                uncer,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            )

        # if count > 100:
        #     break
        count += 1
