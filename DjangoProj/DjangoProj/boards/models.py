from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class Patient(models.Model):
	patient_name = models.CharField(max_length=50)
	DOB = models.DateTimeField()
	age = models.PositiveIntegerField()
	address = models.TextField(max_length=400)
	occupation = models.CharField(max_length=30)
	marital_status = models.PositiveIntegerField()

class Doctor(models.Model):
	name = models.CharField(max_length=30)
	username = models.CharField(max_length=20, unique=True)
	pin = models.PositiveIntegerField(unique=True)