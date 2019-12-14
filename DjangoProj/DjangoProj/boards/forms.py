from django import forms
from boards.models import Examination

class ExamForm(forms.ModelForm):
	age = forms.IntegerField()
	sex = forms.CharField()
	chest_pain = forms.CharField()
	blood_systolic = forms.FloatField()
	blood_diastolic = forms.FloatField()
	chol_overall = forms.FloatField()
	smoke_per_day = forms.IntegerField()
	smoker_years = forms.IntegerField()
	fasting_glucose = forms.FloatField()
	hist_diabetes = forms.BooleanField()
	hist_heart_disease = forms.BooleanField()
	heart_rate = forms.IntegerField()
	exerc_angina = forms.BooleanField()
	
	class Meta:
		model = Examination
		fields = ('age',)