import os
import sys
import cv2
import numpy as np
import math
import xml.dom.minidom
import random
from functools import partial

from pascal_voc_writer import Writer

sys.path.insert(0, "/home/devuser/Documents/workspace/video_toolkit/")

from VideoToolkit.bbox_ops import expand_bbox, convert_bbox


def display(image):
    cv2.imshow("bg1", image)
    keyboard = cv2.waitKey(1000)
    if keyboard == "q" or keyboard == 27:
        breakeyboard = cv2.waav


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def read_annotaions(xml_path, with_size=False):
    res = []

    dom = xml.dom.minidom.parse(xml_path)
    # root = dom.documentElement
    objects = dom.getElementsByTagName("object")
    for obj in objects:
        bndbox = obj.getElementsByTagName("bndbox")[0]
        xmin = bndbox.getElementsByTagName("xmin")[0]
        ymin = bndbox.getElementsByTagName("ymin")[0]
        xmax = bndbox.getElementsByTagName("xmax")[0]
        ymax = bndbox.getElementsByTagName("ymax")[0]
        xmin_data = xmin.childNodes[0].data
        ymin_data = ymin.childNodes[0].data
        xmax_data = xmax.childNodes[0].data
        ymax_data = ymax.childNodes[0].data
        res.append([int(xmin_data), int(ymin_data), int(xmax_data), int(ymax_data)])

    if with_size:
        size = (
            int(dom.getElementsByTagName("width")[0].childNodes[0].data),
            int(dom.getElementsByTagName("height")[0].childNodes[0].data),
        )
        return res, size

    return res


def get_region1d(total_len, patch_len, point, segment_len):
    """gives a region (min, max) for one dimention

    Parameters
    ----------
    total_len : int
        total length possible in this dimention
    patch_len : int
        length of the patch segment
    point : int
        starting point of the segment
    segment_len : int
        segment length

    Returns
    -------
    tuple
        minimun and maximum value of range
    """
    assert total_len >= segment_len
    assert point <= (total_len - segment_len)
    assert total_len >= patch_len

    maxim = min(total_len - patch_len, point)
    minim = max(0, point - patch_len + min(patch_len, segment_len))
    return (minim, maxim)


def get_region2d(img_size, patch_size, bbox):
    """gives a region for x and y that top left corner
    of patch can be picked from

    Parameters
    ----------
    img_size : tuple
        width and height of the original image
    patch_size : tuple
        width and height of patch
    bbox : list
        a list of (x,y, x_prime, y_prime) for annotation (bounding box)

    Returns
    -------
    tuple
    """
    x_region = get_region1d(img_size[0], patch_size[0], bbox[0], bbox[2] - bbox[0])
    y_region = get_region1d(img_size[1], patch_size[1], bbox[1], bbox[3] - bbox[1])
    return x_region, y_region


def is_in_range(top, x, bottom):
    return top <= x and x <= bottom


def get_annotations_in_box(annotation_list, box):
    res_annotations = []
    for annotation in annotation_list:
        ann = None
        if is_in_range(box[0], annotation[0], box[2]):
            if is_in_range(box[1], annotation[1], box[3]):
                ann = [
                    annotation[0],
                    annotation[1],
                    min(annotation[2], box[2]),
                    min(annotation[3], box[3]),
                ]
            elif is_in_range(box[1], annotation[3], box[3]):
                ann = [
                    annotation[0],
                    max(annotation[1], box[1]),
                    min(annotation[2], box[2]),
                    annotation[3],
                ]
        elif is_in_range(box[0], annotation[2], box[2]):
            if is_in_range(box[1], annotation[1], box[3]):
                ann = [
                    max(annotation[0], box[0]),
                    annotation[1],
                    annotation[2],
                    min(annotation[3], box[3]),
                ]
            elif is_in_range(box[3], annotation[3], box[3]):
                ann = [
                    max(annotation[0], box[0]),
                    max(annotation[1], box[1]),
                    annotation[2],
                    annotation[3],
                ]
        if ann is not None:
            res_annotations.append(ann)

    return res_annotations


def transport_box(box, origin_point):
    return [
        box[0] - origin_point[0],
        box[1] - origin_point[1],
        box[2] - origin_point[0],
        box[3] - origin_point[1],
    ]


if __name__ == "__main__":
    base_dir = "/home/devuser/Documents/workspace/data/fully_labeled_1"
    annotations_dir = os.path.join(base_dir, "ann")
    images_dir = os.path.join(base_dir, "data")
    save_dir = "/home/devuser/Documents/workspace/data/segmented_detection"
    patch_size = (156, 156)

    create_directory(os.path.join(save_dir, "data"))
    create_directory(os.path.join(save_dir, "ann"))

    for ann_f in os.listdir(annotations_dir):
        annotation_list, image_size = read_annotaions(
            os.path.join(annotations_dir, ann_f), with_size=True
        )
        name = ann_f[:-4]
        img = cv2.imread(os.path.join(images_dir, name + ".png"))
        count = 0

        for annotation in annotation_list:
            count += 1
            x_region, y_region = get_region2d(image_size, patch_size, annotation)
            min_point = [random.randint(*x_region), random.randint(*y_region)]
            max_point = list(map(lambda x1, x2: x1 + x2, min_point, patch_size))
            patch_box = min_point + max_point

            patch_ann_list = get_annotations_in_box(annotation_list, patch_box)
            patch_ann_list = list(
                map(partial(transport_box, origin_point=min_point), patch_ann_list)
            )

            print(patch_box)
            print(patch_ann_list)
            print(annotation_list)
            print("")

            patched_res = img.copy()[
                patch_box[1] : patch_box[3], patch_box[0] : patch_box[2]
            ]

            # Save annotatioins and images
            res_img_path = os.path.join(save_dir, f"data/{name}_{count}.png")
            cv2.imwrite(
                res_img_path,
                patched_res,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            )
            ann_writer = Writer(
                res_img_path, *patch_size, depth=3, database="Unknown", segmented=0
            )
            for box in patch_ann_list:
                ann_writer.addObject("bird", *box)
            ann_writer.save(os.path.join(save_dir, f"ann/{name}_{count}.xml"))

            # Display annotations
            for box in patch_ann_list:
                cv2.rectangle(
                    patched_res, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2
                )

            display(patched_res)
