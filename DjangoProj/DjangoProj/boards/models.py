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
		fields = (
			self.patient_name, self.age, self.address,
			self.occupation, self.marital_status
		)
		return str(fields)



class Family(models.Model):
	family_name = models.CharField(max_length=30)
	family_hist = models.CharField(max_length=100, null=True)

	fields = (
			self.family_name, self.family_hist
		)
		return str(fields)



class Visit(models.Model):
	doctor = models.ForeignKey('Doctor', on_delete=models.CASCADE)
	patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
	date = models.DateTimeField(auto_now_add=True)
	patient_symptoms = models.CharField(max_length=300, null=True)
	doctor_notes = models.CharField(max_length=100, null=True)
	outcome = models.CharField(max_length=50)

	def __str__(self):
		fields = (
			self.date, self.patient_symptoms, 
			self.doctor_notes, self.outcome
		)
		return str(fields)



class Ailment(models.Model):
	patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
	description = models.CharField(max_length=50)

	def __str__(self):
		return self.description



class Reminders(models.Model):
	patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
	rem_date = models.DateTimeField()
	location = models.CharField(max_length=50, null=True)
	message = models.CharField(max_length=150)

	def __str__(self):
		fields = (
			self.rem_date, self.location, self.message
		)
		return str(fields)



class Allergy(models.Model):
	patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
	medication = models.ForeignKey('Medication', on_delete=models.CASCADE, null=True)
	name = models.CharField(max_length=50)

	def __str__(self):
		return self.name



class Doctor(models.Model):
	name = models.CharField(max_length=30)
	username = models.CharField(max_length=20, unique=True)
	pin = models.PositiveIntegerField(unique=True)

	def __str__(self):
		fields = (
			self.name, self.username, self.pin
		)
		return str(fields)



class Examination(models.Model):
	age = models.PositiveIntegerField(default=0)
	sex = models.PositiveIntegerField()
	chest_pain = models.PositiveIntegerField()
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
		fields = (
			self.age, self.sex, self.chest_pain, self.blood_systolic,
			self.blood_diastolic, self.chol_overall, self.smoke_per_day, 
			self.smoker_years, self.fasting_glucose, self.hist_diabetes,
			self.hist_heart_disease, self.heart_rate, self.exerc_angina
		)
		return str(fields)
