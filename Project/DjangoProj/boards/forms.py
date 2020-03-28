from django import forms
from boards.models import Examination, Patient, Investigation, Visit
import datetime
from dal import autocomplete

GENDER_CHOICES = (
	(1, 'Male'),
	(0, 'Female')
)

CHEST_PAIN_CHOICES = (
	(0, 'None'),
	(1, 'Typical Angina'),
	(2, 'Atypical Angina'),
	(3, 'Non-anginal pain'),
	(4, 'Asymptomatic')
)

VISIT_REASONS = (
	('Examination', 'Examination'),
	('Renew prescription', 'Renew prescription'),
	('Collect prescription', 'Collect prescription')
	# ('', ''),
)

FURTHER_ACTIONS = (
	('None', 'None'),
	('Referral', 'Referral'),
	('Follow up appointment', 'Follow up appointment'),
	('Follow up phone call', 'Follow up phone call')
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
	patient_name = forms.ModelChoiceField(
		queryset=Patient.objects.all(),
		# widget=autocomplete.ModelSelect2(url='patient-autocomplete')
	)

	# class Meta:
	# 	model = Patient
	# 	fields = ('patient_name',)


class CreateVisitForm(forms.ModelForm):
	reason = forms.ChoiceField(choices=VISIT_REASONS, widget=forms.Select(attrs={'class': 'input_field'}))
	date = forms.CharField(widget=forms.widgets.DateTimeInput(attrs={'value': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M"), 'type': 'datetime-local', 'class': 'input_field', 'min': datetime.datetime.today().strftime("%Y-%m-%dT%H:%M")}))

	class Meta:
		model = Visit
		fields = ('reason', 'date',)


class FurtherActionsForm(forms.ModelForm):
	further_actions = forms.ChoiceField(choices=FURTHER_ACTIONS, widget=forms.Select(attrs={'class': 'form-control', 'onchange': 'changeOption()'}))
	ref_to = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control', 'value': None, 'placeholder': 'Refer To..'}))
	ref_reason = forms.CharField(widget=forms.TextInput(attrs={'class': 'form-control', 'value': None, 'placeholder': 'Referral Reason'}))

	class Meta:
		model = Investigation
		fields = ('further_actions', 'ref_to', 'ref_reason',)


class CreatePatientForm(forms.ModelForm):
	patient_name = forms.CharField(label='Patient name', widget=forms.TextInput(attrs={'class': 'input_field'}))
	DOB = forms.CharField(initial=datetime.date.today, widget=forms.widgets.DateTimeInput(attrs={'type': 'date', 'class': 'input_field', 'min': datetime.datetime.today().strftime("%Y-%m-%dT%H:%M")}))
	sex = forms.ChoiceField(choices=GENDER_CHOICES, widget=forms.Select(attrs={'class': 'input_field', }))
	address_line1 = forms.CharField(label='Address line 1', widget=forms.TextInput(attrs={'class': 'input_field'}))
	address_line2 = forms.CharField(label='Address line 2', widget=forms.TextInput(attrs={'class': 'input_field'}))
	address_line3 = forms.CharField(label='Address line 3', widget=forms.TextInput(attrs={'class': 'input_field'}))
	occupation = forms.CharField(label='Occupation', widget=forms.TextInput(attrs={'class': 'input_field'}))
	marital_status = forms.ChoiceField(choices=MARITAL_STATUS, widget=forms.Select(attrs={'class': 'input_field'}))
	tel_num = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'input_field'}))
	home_num = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'input_field'})) 


	class Meta:
		model = Patient
		fields = (
			'patient_name', 'DOB', 'sex',
			'address_line1', 'address_line2', 'address_line3',
			'occupation', 'marital_status',	'tel_num', 'home_num',
		)

class ExamForm(forms.ModelForm):
	height = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'input_field', 'min': 0}))
	weight = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'input_field', 'min': 0}))
	reg_pulse = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.Select(attrs={'class': 'input_field'}))
	cp = forms.ChoiceField(choices=CHEST_PAIN_CHOICES, widget=forms.Select(attrs={'class': 'input_field'}))
	breathlessness = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.Select(attrs={'class': 'input_field'}))
	hdl_chol = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'input_field', 'min': 0}))
	ldl_chol = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'input_field', 'min': 0}))
	triglyceride = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'input_field', 'min': 0}))
	uric_acid = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'input_field', 'min': 0}))
	
	heart_rate = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'input_field', 'min': 0}))
	smoke_per_day = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'input_field', 'min': 0}))
	smoker_years = forms.IntegerField(widget=forms.NumberInput(attrs={'class': 'input_field', 'min': 0}))
	blood_systolic = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'input_field', 'min': 0}))
	blood_diastolic = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'input_field', 'min': 0}))
	chol_total = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'input_field', 'min': 0}))
	fasting_glucose = forms.FloatField(widget=forms.NumberInput(attrs={'class': 'input_field', 'min': 0}))

	
	class Meta:
		model = Examination
		fields = (
			'height', 'weight',
			'reg_pulse', 'cp', 'hdl_chol', 'ldl_chol', 
			'triglyceride', 'uric_acid', 'blood_systolic',
			'blood_diastolic', 'chol_total','heart_rate',
			'smoke_per_day', 'smoker_years', 'fasting_glucose', 
		)
