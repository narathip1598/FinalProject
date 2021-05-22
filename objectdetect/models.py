from django.db import models

class ImageUpload(models.Model):
    photo = models.ImageField()

  