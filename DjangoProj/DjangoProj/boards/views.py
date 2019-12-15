from django.shortcuts import render, redirect
# from django.http import HttpResponse
# from .models import Patient
# from django.settings import BASEDIR
from django.views.generic import TemplateView
from boards.forms import ExamForm
from boards.models import Examination

# For Machine learning model
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# # Create your views here.
# def login(request):
# 	patients = Patient.objects.all()
	
# 	return render(request, 'login.html', {'patients': patients})

class exam(TemplateView):
	template_name = 'exam.html'

	def get(self, request):
		form = ExamForm()
		# exams = Examination.objects.all()
		return render(request, self.template_name, {'form': form})

	def post(self, request):
		form = ExamForm(request.POST)
		if form.is_valid():

			# Save data to  model
			exam = form.save(commit=False)
			exam.user = request.user
			exam.save()
			exam_input = form.cleaned_data
			return redirect('results')
			
			# To remove the value from the input box after submitting
			form = ExamForm()

		args = {'form': form, 'exam_input': exam_input}
		return render(request, self.template_name, args)

class results(TemplateView):
	template_name = 'results.html'
	
	def get_exams(self):
		exams = [Examination.objects.latest('date')]
		return exams

	def get(self, request):
		exams = self.get_exams()

		# TODO
		# Read csv file
		current_dir =  os.path.abspath(os.path.dirname(__file__))
		parent_dir = os.path.abspath(current_dir + "/../")
		pathHeart = parent_dir + '/Data/new_cleveland.csv'
		heart = pd.read_csv(pathHeart)
		print(heart.head())

		# Build classifier
		# Use dummy columns for the categorical features
		# heart.replace(to_replace = -9, value = np.NaN, inplace = True)
		heart = pd.get_dummies(heart, columns = ['sex', 'cp', 'fbs', 'dm', 'famhist', 'exang'])
		print(exams)
		columns_to_scale = ['age', 'trestbps', 'chol', 'cigs', 'years', 'thalrest', 'trestbpd']
		# columns_to_scale = heart.columns
		# print(columns_to_scale)
		standardScaler = StandardScaler()
		heart[columns_to_scale] = standardScaler.fit_transform(heart[columns_to_scale])

		H = heart['target']
		X = heart.drop(['target'], axis = 1)
		X_train, X_test, H_train, H_test = train_test_split(X, H, test_size = 0.01, random_state = 0)

		# KNN
		knn_classifier = KNeighborsClassifier(n_neighbors = 22)
		knn_classifier.fit(X_train, H_train)

		# Predict heart disease


		# Pass prediction 



		return render(request, self.template_name, {'exams':exams})






# def upload_csv(request):
# 	data = {}
# 	if "GET" == request.method:
# 		return render(request, "myapp/upload_csv.html", data)
#     # if not GET, then proceed
# 	try:
# 		csv_file = request.FILES["csv_file"]
# 		if not csv_file.name.endswith('.csv'):
# 			messages.error(request,'File is not CSV type')
# 			return HttpResponseRedirect(reverse("myapp:upload_csv"))
#         #if file is too large, return
# 		if csv_file.multiple_chunks():
# 			messages.error(request,"Uploaded file is too big (%.2f MB)." % (csv_file.size/(1000*1000),))
# 			return HttpResponseRedirect(reverse("myapp:upload_csv"))

# 		file_data = csv_file.read().decode("utf-8")		

# 		lines = file_data.split("\n")
# 		#loop over the lines and save them in db. If error , store as string and then display
# 		for line in lines:						
# 			fields = line.split(",")
# 			data_dict = {}
# 			data_dict["name"] = fields[0]
# 			data_dict["start_date_time"] = fields[1]
# 			data_dict["end_date_time"] = fields[2]
# 			data_dict["notes"] = fields[3]
# 			try:
# 				form = EventsForm(data_dict)
# 				if form.is_valid():
# 					form.save()					
# 				else:
# 					logging.getLogger("error_logger").error(form.errors.as_json())												
# 			except Exception as e:
# 				logging.getLogger("error_logger").error(repr(e))					
# 				pass

# 	except Exception as e:
# 		logging.getLogger("error_logger").error("Unable to upload file. "+repr(e))
# 		messages.error(request,"Unable to upload file. "+repr(e))

# 	return HttpResponseRedirect(reverse("myapp:upload_csv"))
