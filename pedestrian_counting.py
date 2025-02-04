#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#----------------------------------------------

# Imports
import tensorflow as tf
# tf.disable_v2_behavior()

# Object detection imports
from utils import backbone
from api import object_counting_api

# input_video = "./input_images_and_videos/pedestrian_survaillance.mp4"

# By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
# detection_graph, category_index = backbone.set_model('centernet_mobilenetv2_fpn_od', 'mscoco_label_map.pbtxt')

is_color_recognition_enabled = False # set it to true for enabling the color prediction for the detected objects
roi = 385 # roi line position
deviation = 1 # the constant that represents the object counting area
custom_object_name = 'Person'
object_counting_api.cumulative_object_counting_x_axis_webcam(is_color_recognition_enabled, roi, deviation, custom_object_name) # counting all the objects

# http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz
# http://download.tensorflow.org/models/object_detection/tf2/20210210/centernet_mobilenetv2fpn_512x512_coco17_od.tar.gz