from django.shortcuts import render
from .image import UploadFileForm
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import PIL
import tempfile
from tensorflow import keras
import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
import os
from django.core.files.storage import FileSystemStorage
from django.http.response import HttpResponse, HttpResponseRedirect


def show_result(request):
  return render(request,'result.html')


data_dir = 'weather'

batch_size = 32
img_height = 180
img_width = 180

train_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_data = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_data.class_names
print(class_names)


AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)


num_classes = 4


def load_img(path):
  img = tf.io.read_file(path)
  img = tf.image.decode_jpeg(img, channels=3)
  return img

def index(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST)
        uploadfile = request.FILES['file']
        fs = FileSystemStorage()
        name = fs.save(uploadfile.name , uploadfile)
        url = fs.url(name)
        path = url[1:]
        model = keras.models.load_model('imageclassificaiton')
        img = tf.keras.preprocessing.image.load_img(
          path, target_size=(img_height, img_width)
        )
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(
          "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )
        return HttpResponse("This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score)))
    else:
        form = UploadFileForm()
    return render(request,'index.html',{'form':form})

