from django import forms
from boards import models

class ExamForm(forms.ModelForm):
	height = forms.FloatField()
	# weight = forms.FloatField()
	# heart_rate = forms.IntegerField()
	# heart_rhythm = forms.IntegerField()
	# oxygen_saturation = forms.FloatField()
	# blood_pressurre = forms.FloatField()
	# urinalysis_glucose = forms.FloatField()
	# urinalysis_blood = forms.FloatField()
	# urinalysis_protein = forms.FloatField()
	# blood_count = forms.FloatField()
	# fasting_glucose = forms.FloatField()
	# chol_overall = forms.FloatField()
	# chol_HDL = forms.FloatField()
	# chol_LDL = forms.FloatField()
	# uric_acid = forms.FloatField()
	# triglyceride = forms.FloatField()
	# smoke_per_day = forms.IntegerField()
	# alcohol_per_week = forms.IntegerField()
	# recreational_drugs = forms.BooleanField()
	# exercise = forms.BooleanField()
	# healthy_diet = forms.BooleanField()
	# p_examined = forms.ForeignKey(Patient, on_delete = forms.CASCADE)
	# doctor = forms.ForeignKey(Doctor, on_delete = forms.CASCADE)

	class Meta:
		model = models.Examination
		fields = ('height',)