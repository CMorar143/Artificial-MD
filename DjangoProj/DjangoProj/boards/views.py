from django.shortcuts import render
from django.http import HttpResponse
from .models import Patient

# Create your views here.
def login(request):
	patients = Patient.objects.all()
	
	return render(request, 'login.html', {'patients': patients})