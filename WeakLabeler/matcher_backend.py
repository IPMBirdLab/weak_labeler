from typing import List
from skimage.measure import compare_ssim
from skimage.transform import resize
from scipy.stats import wasserstein_distance
import numpy as np
import cv2

from VideoToolkit.bbox_ops import crop_patch
import math


# Implementations are originally from: https://gist.github.com/duhaime/211365edaddf7ff89c0a36d9f3f7956c
# with slight modifications

def _normalize_exposure(img):
  '''
  Normalize the exposure of an image.
  '''
  img = img.astype(int)
  hist = get_histogram(img)
  # get the sum of vals accumulated by each position in hist
  cdf = np.array([sum(hist[:i+1]) for i in range(len(hist))])
  # determine the normalization values for each unit of the cdf
  sk = np.uint8(255 * cdf)
  # normalize each position in the output image
  height, width = img.shape
  normalized = np.zeros_like(img)
  for i in range(0, height):
    for j in range(0, width):
      normalized[i, j] = sk[img[i, j]]
  return normalized.astype(int)


def _get_histogram(img):
  '''
  Get the histogram of an image. For an 8-bit, grayscale image, the
  histogram will be a 256 unit vector in which the nth value indicates
  the percent of the pixels in the image with the given darkness level.
  The histogram's values sum to 1.
  '''
  h, w = img.shape
  hist = [0.0] * 256
  for i in range(h):
    for j in range(w):
      hist[img[i, j]] += 1
  return np.array(hist) / (h * w) 


def earth_movers_distance(img_a, img_b):
  '''
  Measure the Earth Mover's distance between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''
  hist_a = get_histogram(img_a)
  hist_b = get_histogram(img_b)

  return wasserstein_distance(hist_a, hist_b)


def structural_sim(img_a, img_b):
  '''
  Measure the structural similarity between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    {float} a float {-1:1} that measures structural similarity
      between the input images
  '''
  sim, diff = compare_ssim(img_a, img_b, full=True)

  return sim


def pixel_sim(img_a, img_b):
  '''
  Measure the pixel-level similarity between two images
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    {float} a float {-1:1} that measures structural similarity
      between the input images
  '''
  return np.sum(np.absolute(img_a - img_b)) / (img_a.shape[0]*img_a.shape[1]) / 255


def sift_sim(img_a, img_b):
  '''
  Use SIFT features to measure image similarity
  @args:
    {str} path_a: the path to an image file
    {str} path_b: the path to an image file
  @returns:
    TODO
  '''
  # initialize the sift feature detector
  orb = cv2.ORB_create()

  # find the keypoints and descriptors with SIFT
  kp_a, desc_a = orb.detectAndCompute(img_a, None)
  kp_b, desc_b = orb.detectAndCompute(img_b, None)

  # initialize the bruteforce matcher
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # match.distance is a float between {0:100} - lower means more similar
  matches = bf.match(desc_a, desc_b)
  similar_regions = [i for i in matches if i.distance < 70]
  if len(matches) == 0:
    return 0
  return len(similar_regions) / len(matches)


class uniqueBox:
    def __init__(self, image : np.ndarray, bboxes : List[tuple]):
        """[summary]

        Parameters
        ----------
        image : np.array
            the original image
        bboxes : list[tuple]
            bboxes in formate of (x1, y1, x2, y2) representing patches 
        """
        self.image = image
        self.bboxes = bboxes


class Matcher:
    def __init__(self, max_frame_distance):
        self.unique_patches = []
        self.frame_passed_per_frame = []
        self.frame_pass_max = max_frame_distance

    def get_unique_patches(self, img: np.ndarray, bboxes: List[tuple]) -> List[tuple]:
        """Compares given patches of current image with previous unique patches
        and saves and returns unique patches of current image. 

        Parameters
        ----------
        img : np.ndarray
            [description]
        bboxes : List[tuple]
            a list of bounding boxes in (x, y, x1, y1) format

        Returns
        -------
        List[tuple]
            a list of bounding boxes in (x, y, x1, y1) format
        """        
        for i, _ in enumerate(self.frame_passed_per_frame):
            self.frame_passed_per_frame[i] += 1
        self._drop_old_frames()

        unique_boxes = self._get_intra_frame_uniques(img, bboxes)

        unique_boxes = self._get_in_frame_uniques(img, unique_boxes)

        if len(unique_boxes) > 0:
            self.unique_patches.append(uniqueBox(img, unique_boxes))
            self.frame_passed_per_frame.append(0)

        return unique_boxes

    def _get_in_frame_uniques(self, img, bboxes):
        unique_boxes = []

        similarity_sets = [[] for _ in bboxes]
        for i, ibox in enumerate(bboxes):
            patch = crop_patch(img, ibox)
            for j, jbox in enumerate(bboxes):
                if i != j:
                    ref_patch = crop_patch(img, jbox)
                    # if self._is_similar_img(patch, ref_patch) or \
                    #    self._is_similar_box(ibox, jbox):
                    if self._is_similar_box(ibox, jbox):
                        similarity_sets[i].append([j])

        set_sizes = [len(s) for s in similarity_sets]
        idxes = np.array(set_sizes).argsort()

        ignored_idxes = []
        for idx in idxes:
            if idx not in ignored_idxes:
                unique_boxes.append(bboxes[idx])
                for i in similarity_sets[idx]:
                    ignored_idxes.append(i)

        return unique_boxes

    def _get_intra_frame_uniques(self, img, bboxes):
        # TODO: it's a bruteforce implementation. needs improvement
        unique_boxes = []
        for box in bboxes:
            patch = crop_patch(img, box)
            similar = False
            for unq in self.unique_patches:
                ref_img = unq.image
                for ref_box in unq.bboxes:
                    # if self._is_similar_box(box, ref_box):
                    #     similar = True
                    #     break

                    ref_patch = crop_patch(ref_img, ref_box)
                    if self._is_similar_img(patch, ref_patch):
                        similar = True
                        break

            if not similar:
                unique_boxes.append(box)
        
        return unique_boxes

    def _is_similar_box(self, box1, box2):
        dist = self._box_shift_distance(box1, box2)
        if dist <= (256/4)*2:
            return True
        return False

    def _is_similar_img(self, img1, img2):
        similarity = sift_sim(img1, img2)
        if similarity >= 0.8:
            return True
        return False

    def _box_shift_distance(self, box, ref_box):
        return math.sqrt( ((box[0] - ref_box[0])**2) + ((box[1] - ref_box[1])**2) )

    def _drop_old_frames(self):
        keep_idxes = np.where(np.array(self.frame_passed_per_frame) <= self.frame_pass_max)[0]

        unp = []
        fppf = []
        for kidx in list(keep_idxes):
            kidx = int(kidx)
            unp.append(self.unique_patches[kidx])
            fppf.append(self.frame_passed_per_frame[kidx])

        self.unique_patches = unp
        self.frame_passed_per_frame = fppf

