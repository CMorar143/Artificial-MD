from django import forms
from boards.models import Examination

GENDER_CHOICES = (
	(1, 'Male'),
	(2, 'Female')
)

CHEST_PAIN_CHOICES = (
	(1, 'Typical Angina'),
	(2, 'Atypical Angina'),
	(3, 'Non-anginal pain'),
	(4, 'Asymptomatic')
)

TRUE_OR_FALSE = (
	(True, 'Yes'),
	(False, 'No')
)

class ExamForm(forms.ModelForm):
	age = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	sex = forms.ChoiceField(choices=GENDER_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
	chest_pain = forms.ChoiceField(choices=CHEST_PAIN_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
	blood_systolic = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	blood_diastolic = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	chol_overall = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	smoke_per_day = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	smoker_years = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	fasting_glucose = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	hist_diabetes = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.Select(attrs={'class': 'form-control'}))
	hist_heart_disease = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.Select(attrs={'class': 'form-control'}))
	heart_rate = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	exerc_angina = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.Select(attrs={'class': 'form-control'}))
	
	class Meta:
		model = Examination
		fields = (
			'age', 'sex', 'chest_pain', 'blood_systolic',
			'blood_diastolic', 'chol_overall', 'smoke_per_day',
			'smoker_years', 'fasting_glucose', 'hist_diabetes',
			'hist_heart_disease', 'heart_rate', 'exerc_angina'
		)