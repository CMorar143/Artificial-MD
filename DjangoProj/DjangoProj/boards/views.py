from django.shortcuts import render
from django.http import HttpResponse
from .models import Patient
from django.views.generic import TemplateView

# Create your views here.
def login(request):
	patients = Patient.objects.all()
	
	return render(request, 'login.html', {'patients': patients})

def exam(request):
	template_name = 'exam.html'
	return render(request, 'exam.html')