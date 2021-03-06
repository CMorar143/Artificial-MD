from django.db import models
from django.contrib.auth.models import User
from django.conf import settings

# Create your models here.
class Patient(models.Model):
	patient_name = models.CharField(max_length=50)
	DOB = models.DateField()
	sex = models.PositiveSmallIntegerField()
	address_line1 = models.CharField(max_length=40, null=True)
	address_line2 = models.CharField(max_length=40, null=True)
	address_line3 = models.CharField(max_length=40, null=True)
	occupation = models.CharField(max_length=30)
	marital_status = models.PositiveSmallIntegerField()
	acc_balance = models.FloatField(default=0)
	tel_num = models.CharField(max_length=15, null=True)
	home_num = models.CharField(max_length=15, null=True)
	recall_period = models.PositiveSmallIntegerField(null=True)

	def __str__(self):
		fields = (
			self.patient_name, self.DOB, self.address_line1,
			self.address_line2, self.address_line3,	self.occupation, 
			self.marital_status, self.acc_balance, self.tel_num, 
			self.home_num, self.recall_period 
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


class Patient_Family(models.Model):
	patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
	family = models.ForeignKey('Family', on_delete=models.CASCADE)



class Visit(models.Model):
	doctor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
	patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
	date = models.DateTimeField()
	app_length = models.PositiveSmallIntegerField(default=15)
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
	name = models.CharField(max_length=50)
	description = models.CharField(max_length=200, null=True)

	def __str__(self):
		fields = (
			self.name, 
			self.description
		)
		return str(fields)


class Patient_Ailment(models.Model):
	patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
	ailment = models.ForeignKey('Ailment', on_delete=models.CASCADE)


class Allergy(models.Model):
	name = models.CharField(max_length=50)

	def __str__(self):
		return self.name


class Patient_Allergy(models.Model):
	patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
	allergy = models.ForeignKey('Allergy', on_delete=models.CASCADE)


class Medication(models.Model):
	name = models.CharField(max_length=50)
	description = models.CharField(max_length=150)

	def __str__(self):
		return self.name


class Patient_Medication(models.Model):
	patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
	medication = models.ForeignKey('Medication', on_delete=models.CASCADE)


# class Ailment_Medication(models.Model):
# 	medication = models.ForeignKey('Medication', on_delete=models.CASCADE)
# 	ailment = models.ForeignKey('Ailment', on_delete=models.CASCADE)


# class Allergy_Medication(models.Model):
# 	medication = models.ForeignKey('Medication', on_delete=models.CASCADE)
# 	allergy = models.ForeignKey('Allergy', on_delete=models.CASCADE)


class Reminder(models.Model):
	patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
	rem_date = models.DateField()
	location = models.CharField(max_length=50, null=True)
	message = models.CharField(max_length=150)

	def __str__(self):
		fields = (
			self.rem_date, self.location, self.message
		)
		return str(fields)



class Medical_history(models.Model):
	patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
	exam = models.ForeignKey('Examination', on_delete=models.CASCADE, null=True)
	date = models.DateTimeField(auto_now_add=True)
	heart_attack = models.BooleanField(default=False)
	angina = models.BooleanField(default=False)
	breathlessness = models.BooleanField(default=False)
	chest_pain = models.PositiveSmallIntegerField(default=0)
	high_chol = models.BooleanField(default=False)
	high_bp = models.BooleanField(default=False)
	hoarseness = models.BooleanField(default=False)
	wheezing = models.BooleanField(default=False)
	sweating = models.BooleanField(default=False)
	diabetes = models.BooleanField(default=False)
	stressed = models.BooleanField(default=False)
	childhood_illness = models.CharField(max_length=20, null=True)

	def __str__(self):
		fields = (
			self.date, self.heart_attack, self.angina, self.breathlessness,
			self.chest_pain, self.high_chol, self.high_bp,
			self.hoarseness, self.wheezing, self.sweating,
			self.diabetes, self.stressed, self.childhood_illness
		)
		return str(fields)



class Investigation(models.Model):
	visit = models.ForeignKey('Visit', on_delete=models.CASCADE)
	further_actions = models.CharField(max_length=150)
	ref_to = models.CharField(max_length=100, null=True)
	ref_reason = models.CharField(max_length=150, null=True)
	result = models.CharField(max_length=200,null=True)

	def __str__(self):
		fields = (
			self.further_actions, self.ref_to, 
			self.ref_reason, self.result
		)
		return str(fields)



class Examination(models.Model):
	visit = models.ForeignKey('Visit', on_delete=models.CASCADE)

	# Diabetes
	height = models.FloatField()
	weight = models.FloatField()
	reg_pulse = models.BooleanField()
	# cp = models.PositiveIntegerField()
	hdl_chol = models.FloatField(null=True)
	ldl_chol = models.FloatField(null=True)
	triglyceride = models.FloatField()
	uric_acid = models.FloatField()

	# Heart disease 
	heart_rate = models.PositiveSmallIntegerField()
	smoke_per_day = models.PositiveSmallIntegerField() 
	smoker_years = models.PositiveSmallIntegerField()

	# Both
	blood_systolic = models.FloatField()
	blood_diastolic = models.FloatField()
	chol_total = models.FloatField()
	fasting_glucose = models.FloatField()
	

	def __str__(self):
		fields = (
			self.height, self.weight, self.reg_pulse, #self.cp,
			self.hdl_chol, self.ldl_chol, self.triglyceride, 
			self.uric_acid, self.heart_rate,
			self.smoke_per_day, self.smoker_years,
			self.blood_systolic, self.blood_diastolic,
			self.chol_total, self.fasting_glucose
		)
		return str(fields)
