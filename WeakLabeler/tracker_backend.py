import skimage.measure
import numpy as np
import cv2

from VideoToolkit.tools import bbox_from_mask, rescal_to_image

from VideoToolkit.bbox_ops import get_iou, get_overlapping, \
    merge_bboxs, arg_bbox_overlapping_sets, expand_bbox, \
    convert_bbox

from .stat_bs import StatBS



def apply_morphological_ops(frame):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    x = cv2.erode(frame, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    x = cv2.dilate(x, kernel, iterations=1)
    kernel = np.ones((10, 10), np.uint8)
    x = cv2.erode(x, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(30,30))
    x = cv2.dilate(x, kernel, iterations=1)

    return x


grabcut_values = dict((
	("Definite Background", cv2.GC_BGD),
	("Probable Background", cv2.GC_PR_BGD),
	("Definite Foreground", cv2.GC_FGD),
	("Probable Foreground", cv2.GC_PR_FGD),
))


class GrabcutRefiner:
    def __init__(self):
        self.bgdModel = np.zeros((1,65),np.float64)
        self.fgdModel = np.zeros((1,65),np.float64)

    def apply(self, image, bbox=None, single_blob_mask=None):
        if bbox is not None:
            x, y, w, h = bbox
            mask = np.zeros(image.shape[:-1], np.uint8)
        else:
            x, y, w, h = bbox_from_mask(single_blob_mask)[0]
            mask = single_blob_mask

        expanded_bbox = expand_bbox(convert_bbox((x, y, w, h), in_fmt='xywh', out_fmt='xyxy'),
                                    (image.shape[1],
                                    image.shape[0]),
                                    padding=10)
        fx, fy, fw, fh = convert_bbox(expanded_bbox, in_fmt='xyxy', out_fmt='xywh')

        if bbox is not None:
            mask[fy:fy+fh, fx:fx+fw], self.bgdModel, self.fgdModel = \
                            cv2.grabCut(image[fy:fy+fh, fx:fx+fw, :],
                                        None,
                                        (x-fx, y-fy, w, h),
                                        self.bgdModel,
                                        self.fgdModel,
                                        5,
                                        cv2.GC_INIT_WITH_RECT)
        else:
            mask[fy:fy+fh, fx:fx+fw], self.bgdModel, self.fgdModel = \
                            cv2.grabCut(image[fy:fy+fh, fx:fx+fw, :],
                                        mask[fy:fy+fh, fx:fx+fw],
                                        (x-fx, y-fy, w, h),
                                        self.bgdModel,
                                        self.fgdModel,
                                        5,
                                        cv2.GC_INIT_WITH_RECT)

        mask = \
            np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),
                     0, 1).astype('uint8')

        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
        # mask = cv2.dilate(mask, kernel, iterations=3)

        return mask
        

class ObjectToTrack:
    def __init__(self, shape=None):
        self.shape = shape
        self.detector = GrabcutRefiner()
        self.tracker = None
        self.tracker_initialized = False
        self.tracker_ID = None

        self.mask = None
        self.bbox = None

    def set_mask(self, mask):
        self.mask = mask
        self.bbox = bbox_from_mask(mask)[0]

    def refine_mask(self, frame_rgb, mode='mask'):
        if mode == 'mask':
            mask = self.detector.apply(frame_rgb, single_blob_mask=self.mask)
        elif mode == 'bbox':
            mask = self.detector.apply(frame_rgb, bbox=self.bbox)

        bbox = bbox_from_mask(mask)
        if len(bbox) > 0 and bbox[0][2] > 10 and bbox[0][3] > 10:
            self.mask = mask
            self.bbox = bbox[0]
            self.tracker_init(frame_rgb)
            return True
        return False

    def merge_mask(self, mask):
        self.mask[np.where(mask == 1)] = 1
        self.bbox = bbox_from_mask(mask)[0]

    def tracker_init(self, frame_rgb):
        if self.tracker_ID is None:
            self.tracker_ID = np.random.randint(1000)
        self.tracker_initialized = True
        self.tracker = cv2.TrackerCSRT_create()
        return self.tracker.init(frame_rgb, self.bbox)

    def tracker_update(self, frame_rgb):
        ok, self.bbox = self.tracker.update(frame_rgb)
        return ok


class MultiObjectTracker:
    def __init__(self, shape):
        self.shape = shape
        self.statbased = StatBS(shape)
        self.objects = []

    def get_bboxs(self):
        return [obj.bbox for obj in self.objects]

    def get_mask(self):
        mask = np.zeros(self.shape, np.uint8)
        for obj in self.objects:
            mask[np.where(obj.mask == 1)] = 1
        return mask

    def set_latest_window_frame(self, frame_rgb):
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

        self.statbased.apply(frame)

    def get_new_bboxs(self, frame_rgb):
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)

        # bg = self.statbased.get_bmask(frame)
        bg, _, _ = self.statbased.get_uncertainty_maps(frame)
        bg = rescal_to_image(bg)  

        mask = apply_morphological_ops(bg).astype('uint8')
        labeled_mask = skimage.measure.label(np.where(mask > 0 , 1, 0)) #, connectivity=2)
        labels = np.unique(labeled_mask)

        resulting_masks = []
        resulting_bboxes = []
        for i in labels[1:]:
            mask = np.where(labeled_mask == i, 1, 0).astype('uint8')
            resulting_masks.append(mask)
            resulting_bboxes.append(bbox_from_mask(mask)[0])

        return resulting_masks, resulting_bboxes

    def update_tracked_bboxs(self, frame_rgb):
        if len(self.objects) == 0:
            return

        failed_idxs = []
        for i, obj in enumerate(self.objects):
            ok = obj.tracker_update(frame_rgb)
            if not ok:
                failed_idxs.append(i)

        failed_idxs.sort(reverse=True)
        for i in failed_idxs:
            print(f"tracker {self.objects[i].tracker_ID} failed.")
            del self.objects[i]

    def merge_overlapping_objects(self, frame_rgb):
        bboxes = self.get_bboxs()
        arg_ol_sts = arg_bbox_overlapping_sets(bboxes, threshold=0.8)
        del_idxs = []
        for s in arg_ol_sts:
            i = s[0]
            for j in s[1:]:
                self.objects[i].merge_mask(self.objects[j].mask)
                self.objects[i].tracker_init(frame_rgb)
                del_idxs.append(i)

        del_idxs.sort(reverse=True)
        for i in del_idxs:
            del self.objects[i]

    def update(self, frame_rgb):
        if len(self.objects) > 0:
            self.update_tracked_bboxs(frame_rgb)
        masks, bboxs = self.get_new_bboxs(frame_rgb)

        # it is important that which new bounding box gets merged with which
        # previous object
        used_masks = [0 for _ in range(len(masks))]
        for obj in self.objects:
            for i, (mask, bbox) in enumerate(zip(masks, bboxs)):
                overlap = get_overlapping(bbox, obj.bbox)
                iou = get_iou(bbox, obj.bbox)
                if overlap > 0.9 or iou > 0.9:
                    obj.set_mask(mask)
                    obj.tracker_init(frame_rgb)
                    used_masks[i] += 1
                if overlap > 0.75 or iou > 0.75:
                    obj.merge_mask(mask)
                    used_masks[i] += 1

        for i, v in enumerate(used_masks):
            if v == 0:
                obj = ObjectToTrack(frame_rgb.shape)
                obj.set_mask(masks[i])
                obj.tracker_init(frame_rgb)
                # if obj.refine_mask(frame_rgb):
                self.objects.append(obj)

        self.merge_overlapping_objects(frame_rgb)
