from django.shortcuts import render, redirect
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

	def post(self, request):
		form = ExamForm(request.POST)
		if form.is_valid():
			# Save data to  model
			# exam = form.save(commit=False)
			form.save()
			# exam.user = request.user
			# exam.save()
			exam_input = form.cleaned_data['height']
			return redirect('exam')
			
			# To remove the value from the input box after submitting
			# form = ExamForm()

		args = {'form': form, 'exam_input': exam_input}
		return render(request, self.template_name, args)