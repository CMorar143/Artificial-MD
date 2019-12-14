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
	age = forms.IntegerField()
	sex = forms.ChoiceField(choices=GENDER_CHOICES, widget=forms.RadioSelect())
	chest_pain = forms.ChoiceField(choices=CHEST_PAIN_CHOICES, widget=forms.RadioSelect())
	blood_systolic = forms.FloatField()
	blood_diastolic = forms.FloatField()
	chol_overall = forms.FloatField()
	smoke_per_day = forms.IntegerField()
	smoker_years = forms.IntegerField()
	fasting_glucose = forms.FloatField()
	hist_diabetes = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.RadioSelect())
	hist_heart_disease = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.RadioSelect())
	heart_rate = forms.IntegerField()
	exerc_angina = forms.ChoiceField(choices=TRUE_OR_FALSE, widget=forms.RadioSelect())
	
	class Meta:
		model = Examination
		fields = (
			'age', 'sex', 'chest_pain', 'blood_systolic', 
			'blood_diastolic', 'chol_overall', 'smoke_per_day', 
			'smoker_years', 'fasting_glucose', 'hist_diabetes',
			'hist_heart_disease', 'heart_rate', 'exerc_angina'
		)