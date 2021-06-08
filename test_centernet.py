# https://towardsdatascience.com/object-detection-by-tensorflow-2-x-e1199558abc
import os
import numpy as np
from PIL import Image
# import matplotlib.pyplot as plt
import warnings
import tensorflow as tf
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings
import cv2
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
# This is required to display the images.
# %matplotlib inline

# print('Running inference for {}... '.format(image_path), end='')

img = cv2.imread('./test_image/pedestrian.jpg')
# cv2.imshow('test', img)
image_np = np.array(img)

category_index = label_map_util.create_category_index_from_labelmap('./centernet_mobilenetv2_fpn_od/mscoco_label_map.pbtxt.txt',  use_display_name=True)

# detect_fn = tf.saved_model.load('./centernet_hg104_1024x1024_coco17_tpu-32/saved_model')
# detect_fn = tf.saved_model.load('./centernet_mobilenetv2_fpn_od/saved_model')
# detect_fn = tf.saved_model.load('./ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model')
detect_fn = tf.saved_model.load('./ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model')


# image_np = np.array(Image.open('./test_image/pedestrian.jpg'))

# Things to try:
# Flip horizontally
# image_np = np.fliplr(image_np).copy()

# Convert image to grayscale
# image_np = np.tile(
#     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image_np)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        min_score_thresh=.30,
        agnostic_mode=False)

cv2.imshow("image", image_np_with_detections)
cv2.waitKey(0) & 0xFF
# plt.figure(figsize = (12,8))
# plt.imshow(image_np_with_detections)
# plt.show()

# centernet_mobilenetv2_fpn_od