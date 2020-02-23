from django import forms
from boards.models import Examination, Patient
import datetime

GENDER_CHOICES = (
	(1, 'Male'),
	(2, 'Female')
)

CHEST_PAIN_CHOICES = (
	(0, 'None'),
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

MARITAL_STATUS = (
	(1, 'Single'),
	(2, 'Married'),
	(3, 'Widowed'),
	(4, 'Divorced')
)

class SelectPatientForm(forms.Form):
	patient_name = forms.ModelChoiceField(queryset=Patient.objects.all())

class CreatePatientForm(forms.ModelForm):
	patient_name = forms.CharField(label='Patient name', widget=forms.TextInput(attrs={'class': 'form-control'}))
	DOB = forms.DateField(initial=datetime.date.today, widget=forms.SelectDateWidget(years=range(1900, 2020), attrs={'class': 'form-datefield'}))
	age = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	sex = forms.ChoiceField(choices=GENDER_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
	address_line1 = forms.CharField(label='Address line 1', widget=forms.TextInput(attrs={'class': 'form-control'}))
	address_line2 = forms.CharField(label='Address line 2', widget=forms.TextInput(attrs={'class': 'form-control'}))
	address_line3 = forms.CharField(label='Address line 3', widget=forms.TextInput(attrs={'class': 'form-control'}))
	occupation = forms.CharField(label='Occupation', widget=forms.TextInput(attrs={'class': 'form-control'}))
	marital_status = forms.ChoiceField(choices=MARITAL_STATUS, widget=forms.Select(attrs={'class': 'form-control'}))
	tel_num = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	home_num = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'})) 

	class Meta:
		model = Patient
		fields = (
			'patient_name', 'DOB', 'age', 'sex',
			'address_line1', 'address_line2', 'address_line3',
			'occupation', 'marital_status',	'tel_num', 'home_num',
		)

class ExamForm(forms.ModelForm):
	height = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	weight = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	reg_pulse = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.Select(attrs={'class': 'form-control'}))
	pulse_type = forms.ChoiceField(choices=PULSE_TYPE_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
	protein = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	hdl_chol = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	ldl_chol = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	triglyceride = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	uric_acid = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	heart_rate = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	smoke_per_day = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	smoker_years = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	blood_systolic = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	blood_diastolic = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	chol_overall = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
	fasting_glucose = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'form-control'}))

	
	class Meta:
		model = Examination
		fields = (
			'height', 'weight',
			'reg_pulse', 'pulse_type', 'protein', 'hdl_chol',
			'ldl_chol', 'triglyceride', 'uric_acid', 'blood_systolic',
			'blood_diastolic', 'chol_overall','heart_rate',
			'smoke_per_day', 'smoker_years', 'fasting_glucose', 
		)




# age = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'form-control'}))
# sex = forms.ChoiceField(choices=GENDER_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
# chest_pain = forms.ChoiceField(choices=CHEST_PAIN_CHOICES, widget=forms.Select(attrs={'class': 'form-control'}))
# hist_diabetes = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.Select(attrs={'class': 'form-control'}))
# hist_heart_disease = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.Select(attrs={'class': 'form-control'}))
# exerc_angina = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.Select(attrs={'class': 'form-control'}))
# fields = (
# # 'age', 'sex', 'chest_pain', 
# 'height', 'weight',
# 'reg_pulse', 'pulse_type', 'protein', 'hdl_chol',
# 'ldl_chol', 'triglyceride', 'uric_acid', 'blood_systolic',
# 'blood_diastolic', 'chol_overall','heart_rate',
# 'smoke_per_day', 'smoker_years', 'fasting_glucose', 
# # 'hist_diabetes', 'hist_heart_disease', 
# #, 'exerc_angina'
# )