from django import forms
from boards.models import Examination

GENDER_CHOICES = (
	('M', 'Male'),
	('F', 'Female')
)

CHEST_PAIN_CHOICES = (
	()
)

class ExamForm(forms.ModelForm):
	age = forms.IntegerField()
	sex = forms.ChoiceField(choices=GENDER_CHOICES, widget=forms.RadioSelect())
	chest_pain = forms.ChoiceField(choices=CHEST_PAIN_CHOICES, widget=forms.RadioSelect())
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
		fields = (
			'age', 'sex', 'chest_pain', 
			'blood_systolic', 'blood_diastolic', 'chol_overall',
			'smoke_per_day', 'smoker_years', 'fasting_glucose',
			'hist_diabetes', 'hist_heart_disease', 'exerc_angina'
		)