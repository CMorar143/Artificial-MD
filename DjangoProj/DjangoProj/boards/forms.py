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

PULSE_TYPE_CHOICES = (
	(1, 'Radial'),
	(2, 'Brachial')
)

TRUE_OR_FALSE = (
	(True, 'Yes'),
	(False, 'No')
)

class PatientForm(forms.ModelForm):
	

class ExamForm(forms.ModelForm):
	age = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	sex = forms.ChoiceField(choices=GENDER_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
	chest_pain = forms.ChoiceField(choices=CHEST_PAIN_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
	
	height = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	weight = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	reg_pulse = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.Select(attrs={'class': 'form-control'}))
	pulse_type = forms.ChoiceField(choices=PULSE_TYPE_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
	protein = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	hdl_chol = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	ldl_chol = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	triglyceride = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	uric_acid = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))

	blood_systolic = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	blood_diastolic = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	chol_overall = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	smoke_per_day = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	smoker_years = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	fasting_glucose = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	# hist_diabetes = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.Select(attrs={'class': 'form-control'}))
	# hist_heart_disease = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.Select(attrs={'class': 'form-control'}))
	heart_rate = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	exerc_angina = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.Select(attrs={'class': 'form-control'}))
	
	class Meta:
		model = Examination
		fields = (
			'age', 'sex', 'chest_pain', 'height', 'weight',
			'reg_pulse', 'pulse_type', 'protein', 'hdl_chol',
			'ldl_chol', 'triglyceride', 'uric_acid', 'blood_systolic',
			'blood_diastolic', 'chol_overall', 'smoke_per_day',
			'smoker_years', 'fasting_glucose', 
			# 'hist_diabetes', 'hist_heart_disease', 
			'heart_rate', 'exerc_angina'
		)