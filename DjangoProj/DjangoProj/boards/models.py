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

class Examination(models.Model)
	age = models.PositiveIntegerField()
	sex = models.CharField()
	chest_pain = models.CharField()

	height = models.FloatField()
	date = models.DateTimeField(auto_now_add=True)
	weight = models.FloatField()
	heart_rate = models.PositiveIntegerField()
	# heart_rhythm = models.PositiveIntegerField()
	oxygen_saturation = models.FloatField()
	blood_pressure = models.FloatField()
	# urinalysis_glucose = models.FloatField()
	# urinalysis_blood = models.FloatField()
	# urinalysis_protein = models.FloatField()
	# blood_count = models.FloatField()
	fasting_glucose = models.FloatField()
	chol_overall = models.FloatField()
	# chol_HDL = models.FloatField()
	# chol_LDL = models.FloatField()
	# uric_acid = models.FloatField()
	# triglyceride = models.FloatField()
	smoker_years = models.PositiveIntegerField()
	smoke_per_day = models.PositiveIntegerField()
	# alcohol_per_week = models.PositiveIntegerField()
	# recreational_drugs = models.BooleanField()
	# exercise = models.BooleanField()
	# healthy_diet = models.BooleanField()
	# user_id = models.FloatField()
	user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete = models.CASCADE)
	# p_examined = models.ForeignKey(Patient, on_delete = models.CASCADE)
	# doctor = models.ForeignKey(Doctor, on_delete = models.CASCADE)

	def __str__(self):
		return str(self.height)
