from django.db import models
from django.contrib.auth.models import User
from django.conf import settings

# Create your models here.
class Patient(models.Model):
	family = models.ForeignKey('Family', on_delete=models.CASCADE, null=True)
	patient_name = models.CharField(max_length=50)
	DOB = models.DateTimeField()
	age = models.PositiveIntegerField()
	sex = models.PositiveIntegerField()
	address_line1 = models.CharField(max_length=40, null=True)
	address_line2 = models.CharField(max_length=40, null=True)
	address_line3 = models.CharField(max_length=40, null=True)
	occupation = models.CharField(max_length=30)
	marital_status = models.PositiveIntegerField()
	acc_balance = models.PositiveIntegerField(default=0)
	tel_num = models.PositiveIntegerField(null=True)
	home_num = models.PositiveIntegerField(null=True)
	next_app = models.DateTimeField(null=True)
	recall_period = models.PositiveIntegerField(null=True)

	def __str__(self):
		fields = (
			self.patient_name, self.DOB, self.age, self.address_line1,
			self.address_line2, self.address_line3,	self.occupation, 
			self.marital_status, self.acc_balance, self.tel_num, 
			self.home_num, self.next_app, self.recall_period 
		)
		return str(self.patient_name)



class Family(models.Model):
	family_name = models.CharField(max_length=30)
	family_hist = models.CharField(max_length=100, null=True)

	def __str__(self):
		fields = (
				self.family_name, self.family_hist
			)
		return str(fields)



class Visit(models.Model):
	doctor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
	patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
	date = models.DateTimeField(auto_now_add=True)
	reason = models.CharField(max_length=300, null=True)
	doctor_notes = models.CharField(max_length=100, null=True)
	outcome = models.CharField(max_length=50, null=True)

	def __str__(self):
		fields = (
			self.date, self.reason, 
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
	patient = models.ForeignKey('Patient', primary_key=True, on_delete=models.CASCADE)
	# exam = models.ForeignKey('Examination', on_delete=models.CASCADE)
	heart_attack = models.BooleanField()
	angina = models.BooleanField()
	breathlessness = models.BooleanField()
	chest_pain = models.PositiveIntegerField()
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



# class Doctor(models.Model):
# 	name = models.CharField(max_length=30)
# 	username = models.CharField(max_length=20, unique=True)
# 	pin = models.PositiveIntegerField(unique=True)

# 	def __str__(self):
# 		fields = (
# 			self.name, self.username, self.pin
# 		)
# 		return str(fields)



class Examination(models.Model):
	visit = models.ForeignKey('Visit', on_delete=models.CASCADE)

	# Diabetes
	height = models.FloatField()
	weight = models.FloatField()
	reg_pulse = models.BooleanField()
	pulse_type = models.PositiveIntegerField()
	protein = models.FloatField()
	hdl_chol = models.FloatField(null=True)
	ldl_chol = models.FloatField(null=True)
	triglyceride = models.FloatField()
	uric_acid = models.FloatField()

	# Heart disease 
	pulse_rate = models.PositiveIntegerField()
	smoke_per_day = models.PositiveIntegerField() 
	smoker_years = models.PositiveIntegerField() 
	# exerc_angina = models.BooleanField()
	

	# Both
	blood_systolic = models.FloatField()
	blood_diastolic = models.FloatField()
	chol_overall = models.FloatField()
	fasting_glucose = models.FloatField()
	

	def __str__(self):
		fields = (
			self.height, self.weight, self.reg_pulse, self.pulse_type,
			self.protein, self.hdl_chol, self.ldl_chol, 
			self.triglyceride, self.uric_acid, self.pulse_rate, 
			self.smoke_per_day, self.smoker_years,
			self.blood_systolic, self.blood_diastolic, 
			self.chol_overall, self.fasting_glucose
		)
		return str(fields)
