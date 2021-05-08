from .models import resize_image,load_img,draw_boxes,display_image
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import tempfile
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import os

# Create your views here.

image_path = "01.jpg"
path = resize_image(image_path, 1280, 856, True)

image = load_img(path)
image_test  = tf.image.convert_image_dtype(image, tf.float32)[tf.newaxis, ...]
module = "https://tfhub.dev/google/faster_rcnn/openimages_v4/inception_resnet_v2/1"

detector = hub.load(module).signatures['default']
result = detector(image_test)
result = {key:value.numpy() for key,value in result.items()}
image_with_boxes = draw_boxes(image.numpy(), result["detection_boxes"], result["detection_class_entities"], result["detection_scores"])
display_image(image_with_boxes)