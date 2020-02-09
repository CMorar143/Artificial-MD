from django.db import models
from django.contrib.auth.models import User
from django.conf import settings

# Create your models here.
class Patient(models.Model):
	family = models.ForeignKey('Family', on_delete=models.CASCADE)
	patient_name = models.CharField(max_length=50)
	DOB = models.DateTimeField()
	age = models.PositiveIntegerField()
	address = models.TextField(max_length=400)
	occupation = models.CharField(max_length=30)
	marital_status = models.PositiveIntegerField()
	acc_balance = models.PositiveIntegerField()
	tel_num = models.PositiveIntegerField(null=True)
	home_num = models.PositiveIntegerField(null=True)
	next_app = models.DateTimeField(null=True)
	recall_period = models.PositiveIntegerField(null=True)

	def __str__(self):
		fields = (
			self.patient_name, self.DOB, self.age, self.address,
			self.occupation, self.marital_status, self.acc_balance,
			self.tel_num, self.home_num, self.next_app, self.recall_period 
		)
		return str(fields)



class Family(models.Model):
	family_name = models.CharField(max_length=30)
	family_hist = models.CharField(max_length=100, null=True)

	def __str__(self):
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
	medication = models.ForeignKey('Medication', on_delete=models.CASCADE, null=True)
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



class Medical_history(models.Model):
	exam = models.ForeignKey('Examination', on_delete=models.CASCADE)
	heart_attack = models.BooleanField()
	angina = models.BooleanField()
	breathlessness = models.BooleanField()
	chest_pain = models.BooleanField()
	high_chol = models.BooleanField()
	high_bp = models.BooleanField()
	hoarseness = models.BooleanField()
	wheezing = models.BooleanField()
	sweating = models.BooleanField()
	diabetes = models.BooleanField()
	stressed = models.BooleanField()
	childhood_illness = models.CharField(max_length=20, null=True)

	def __str__(self):
		fields = (
			self.heart_attack, self.angina, self.breathlessness,
			self.chest_pain, self.high_chol, self.high_bp,
			self.hoarseness, self.wheezing, self.sweating,
			self.diabetes, self.stressed, self.childhood_illness
		)
		return str(fields)



class Investigation(models.Model):
	visit = models.ForeignKey('Visit', on_delete=models.CASCADE)
	date = models.DateTimeField(auto_now_add=True)
	further_actions = models.CharField(max_length=150)
	ref_to = models.CharField(max_length=100, null=True)
	ref_reason = models.CharField(max_length=150, null=True)
	result = models.CharField(max_length=200)

	def __str__(self):
		fields = (
			self.date, self.further_actions, self.ref_to, 
			self.ref_reason, self.result
		)
		return str(fields)



class Medication(models.Model):
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
	height = models.FloatField()
	weight = models.FloatField()
	heart_rate = models.PositiveIntegerField()
	heart_rhythm = models.PositiveIntegerField()
	oxygen = models.FloatField()

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
