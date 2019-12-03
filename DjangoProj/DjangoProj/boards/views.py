from django.shortcuts import render
from django.http import HttpResponse
from .models import Patient

# Create your views here.
def home(request):
	patients = Patient.objects.all()
	patient_names = list()

	for patient in patients:
		patient_names.append(patient.patient_name)

	response_html = 'Patient Names are: <br>' + '<br>'.join(patient_names)
	return HttpResponse(response_html)