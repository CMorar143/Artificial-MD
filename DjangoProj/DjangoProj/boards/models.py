from django.db import models
from django.contrib.auth.models import User
from django.conf import settings

# Create your models here.
class Patient(models.Model):
	patient_name = models.CharField(max_length=50)
	# DOB = models.DateTimeField()
	age = models.PositiveIntegerField()
	address = models.TextField(max_length=400)
	occupation = models.CharField(max_length=30)
	marital_status = models.PositiveIntegerField()

	def __str__(self):
		return self.patient_name

class Doctor(models.Model):
	name = models.CharField(max_length=30)
	username = models.CharField(max_length=20, unique=True)
	pin = models.PositiveIntegerField(unique=True)

	def __str__(self):
		return self.username

class Examination(models.Model):
	age = models.PositiveIntegerField(default=0)
	sex = models.CharField(max_length=6)
	chest_pain = models.CharField(max_length=50)
	blood_systolic = models.FloatField()
	blood_diastolic = models.FloatField()
	chol_overall = models.FloatField()
	smoke_per_day = models.PositiveIntegerField()
	smoker_years = models.PositiveIntegerField()
	fasting_glucose = models.FloatField()
	hist_diabetes = models.BooleanField()
	hist_heart_disease = models.BooleanField()
	heart_rate = models.PositiveIntegerField()
	exerc_angina = models.BooleanField()
	date = models.DateTimeField(auto_now_add=True)
	user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete = models.CASCADE)

	def __str__(self):
		return str(self.height)
