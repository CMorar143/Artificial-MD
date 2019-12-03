from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Patient(models.Model):
	patient_name = models.CharField(max_length=50)
	DOB = models.DateTimeField()
	age = models.PositiveIntegerField()
	address = models.TextField()
	occupation = models.CharField(max_length=30)
	marital_status = models.PositiveIntegerField()