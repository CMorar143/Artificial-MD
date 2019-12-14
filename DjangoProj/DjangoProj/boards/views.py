from django.shortcuts import render
# from django.http import HttpResponse
# from .models import Patient
from django.views.generic import TemplateView
from boards.forms import ExamForm

# # Create your views here.
# def login(request):
# 	patients = Patient.objects.all()
	
# 	return render(request, 'login.html', {'patients': patients})

class exam(TemplateView):
	template_name = 'exam.html'

	def get(self, request):
		form = ExamForm()
		return render(request, self.template_name, {'form': form})