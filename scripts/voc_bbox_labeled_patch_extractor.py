import os
import sys
import cv2
import numpy as np
import re
import xml.dom.minidom
import random
from functools import partial

from pascal_voc_writer import Writer

sys.path.insert(0, "/workspace8/video_toolkit")

from VideoToolkit.bbox_ops import expand_bbox, convert_bbox, get_iou, get_overlapping


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
        res.append(
            [
                int(float(xmin_data)),
                int(float(ymin_data)),
                int(float(xmax_data)),
                int(float(ymax_data)),
            ]
        )

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
    return top < x and x < bottom


def get_annotations_in_box(annotation_list, box, indices=False):
    if indices:
        indice_list = []
    res_annotations = []
    for idx, annotation in enumerate(annotation_list):
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
            if indices:
                assert annotation_list[idx] == annotation
                indice_list.append(idx)
    if indices:
        return res_annotations, indice_list
    return res_annotations


def transport_box(box, origin_point):
    return [
        box[0] - origin_point[0],
        box[1] - origin_point[1],
        box[2] - origin_point[0],
        box[3] - origin_point[1],
    ]


def drop_bad_boxes(bbox_list, original_bbox_list, min_dim=10):
    keep = [True for _ in range(len(bbox_list))]
    for i, (bbox, obbox) in enumerate(zip(bbox_list, original_bbox_list)):
        # if abs(bbox[2] - bbox[0]) < 10 and abs(bbox[3] - bbox[1]) < 10:
        #     keep[i] = False
        iou = get_iou(
            convert_bbox(bbox, "xyxy", "xywh"), convert_bbox(obbox, "xyxy", "xywh")
        )
        if iou < 0.4:
            keep[i] = False
    res_list = [bbox for i, bbox in enumerate(bbox_list) if keep[i]]
    return res_list


if __name__ == "__main__":
    base_dir = "/workspace8/data/labeled_data"
    annotations_dir = os.path.join(base_dir, "train")
    images_dir = os.path.join(base_dir, "train")
    save_dir = "/workspace8/dataset/segmented_detection/train"
    patch_size = (256, 256)

    create_directory(os.path.join(save_dir, "data"))
    create_directory(os.path.join(save_dir, "ann"))
    create_directory(os.path.join(save_dir, "objness"))
    create_directory(os.path.join(save_dir, "vis"))

    for img_f in [
        f for f in os.listdir(images_dir) if re.search(r"exp[0-9]*_[0-9]*.png$", f)
    ]:
        name = img_f[:-4]
        annotation_list, image_size = read_annotaions(
            os.path.join(annotations_dir, name + ".xml"), with_size=True
        )

        img = cv2.imread(os.path.join(images_dir, name + ".png"))
        objness = cv2.imread(os.path.join(images_dir, name + "_obn.png"))
        count = 0

        for annotation in annotation_list:
            count += 1
            x_region, y_region = get_region2d(image_size, patch_size, annotation)
            min_point = [random.randint(*x_region), random.randint(*y_region)]
            max_point = list(map(lambda x1, x2: x1 + x2, min_point, patch_size))
            patch_box = min_point + max_point

            patch_ann_list, patch_ann_indice_list = get_annotations_in_box(
                annotation_list, patch_box, indices=True
            )
            patch_ann_list = drop_bad_boxes(
                patch_ann_list, [annotation_list[i] for i in patch_ann_indice_list]
            )
            patch_ann_list = list(
                map(partial(transport_box, origin_point=min_point), patch_ann_list)
            )
            if not len(patch_ann_list) > 0 or np.array(patch_ann_list).shape[1] != 4:
                print(np.array(patch_ann_list).shape)
                print(name)
                print(count)
                continue

            # print(patch_box)
            # print(np.array(patch_ann_list).shape)
            # print(annotation_list)
            # print("")

            patched_res = img.copy()[
                patch_box[1] : patch_box[3], patch_box[0] : patch_box[2]
            ]

            patched_objness_res = objness.copy()[
                patch_box[1] : patch_box[3], patch_box[0] : patch_box[2]
            ]

            # Save image and objectness
            res_img_path = os.path.join(save_dir, f"data/{name}_{count}.png")
            cv2.imwrite(
                res_img_path,
                patched_res,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            )
            res_obn_path = os.path.join(save_dir, f"objness/{name}_{count}.png")
            cv2.imwrite(
                res_obn_path,
                patched_objness_res,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            )

            # Save annotatioins
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

            # display(patched_res)

            # Save Displayed annotations
            res_img_path = os.path.join(save_dir, f"vis/{name}_{count}.png")
            cv2.imwrite(
                res_img_path,
                patched_res,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            )
