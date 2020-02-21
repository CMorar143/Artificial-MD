from django.shortcuts import render, redirect
from django.urls import reverse
from urllib.parse import urlencode
from django.conf import settings
# from django.http import HttpResponse
# from django.settings import BASEDIR
from django.views.generic import TemplateView
from boards.forms import ExamForm, CreatePatientForm, SelectPatientForm
from boards.models import Examination, Patient, Visit, Medical_history

# For Machine learning model
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

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
			p_name = request.GET.get('patient')
			patient = Patient.objects.get(patient_name=p_name)
			visit = Visit.objects.get(patient=patient).latest('date')
			exam = form.save(commit=False)
			exam.user = request.user
			exam.visit = visit
			exam.save()
			exam_input = form.cleaned_data
			return redirect('results')
			
			# To remove the value from the input box after submitting
			# form = ExamForm()

		args = {'form': form, 'exam_input': exam_input}
		return render(request, self.template_name, args)



class test(TemplateView):
	template_name = 'test.html'
	
	def get(self, request):
		args = {}
		patient = request.GET.get('patient')
		args['patient'] = patient

		return render(request, self.template_name, args)



class patient(TemplateView):
	template_name = 'patient.html'

	def get(self, request):
		patients = Patient.objects.all()
		Createform = CreatePatientForm()
		Selectform = SelectPatientForm()
		return render(request, self.template_name, {'patients': patients, 'Createform': Createform, 'Selectform': Selectform})

	def post(self, request):
		if 'create_patient' in request.POST:
			Createform = CreatePatientForm(request.POST)
			if Createform.is_valid():

				# Save data to model
				patient = Createform.save(commit=False)
				patient.user = request.user
				patient.save()
				patient_input = Createform.cleaned_data
				return redirect('exam')
				
				# To remove the value from the input box after submitting
				# form = CreatePatientForm()

			args = {'Createform': Createform, 'Selectform': Selectform, 'exam_input': patient_input}
			return render(request, self.template_name, args)

		elif 'select_patient' in request.POST:
			Selectform = SelectPatientForm(request.POST)
			if Selectform.is_valid():
				args = {}
				patient = Selectform.cleaned_data['patient_name']
				args['patient'] = patient
				p = Patient.objects.get(patient_name=patient)

				visit = Visit.objects.create(doctor=request.user, patient=p, reason="For examination")

				base_url = reverse('exam')
				query_string = urlencode(args)
				url = '{}?{}'.format(base_url, query_string)
				return redirect(url)
		# return render(request, 'test.html', {"patient" : patient})




class results(TemplateView):
	template_name = 'results.html'
	
	def get_data(self):
		p_name = request.GET.get('patient')

		patient = Patient.objects.get(patient_name=p_name)
		visit = Visit.objects.get(patient=patient).latest('date')
		exams = Examination.objects.get(visit=visit)
		med_hist = Medical_history.objects.get(patient=patient)

		exam_vals = exams.values()
		patient_vals = patient.values('age', 'sex')
		med_hist_vals = med_hist.values(
			'heart_attack', 'angina', 'breathlessness',
			'chest_pain', 'high_chol', 'high_bp',
			'diabates'
			)

		# Extract heart data
		heart_vals = []


		# Extract diabates data
		diabetes_vals = []


		return heart_vals, diabetes_vals


	def load_dataframe(self):
		# Load heart disease dataset into pandas dataframe
		current_dir =  os.path.abspath(os.path.dirname(__file__))
		parent_dir = os.path.abspath(current_dir + "/../")
		pathHeart = parent_dir + '/Data/new_cleveland.csv'
		heart = pd.read_csv(pathHeart)

		return heart


	def scale_values(self, heart, standardScaler):
		heart = pd.get_dummies(heart, columns = ['cp'])
		columns_to_scale = ['age', 'trestbps', 'chol', 'cigs', 'years', 'thalrest', 'trestbpd']
		heart[columns_to_scale] = standardScaler.fit_transform(heart[columns_to_scale])

		return heart


	def split_dataset(self, X, D):
		return train_test_split(X, D, test_size = 0.01, random_state = 0)


	def KNN(self, X_train, H_train):
		knn_classifier = KNeighborsClassifier(n_neighbors = 9)
		knn_classifier.fit(X_train, H_train)

		return knn_classifier


	def decision_tree(self, X_train, H_train):
		dt_classifier = DecisionTreeClassifier(max_features = 10, random_state = 0)
		dt_classifier.fit(X_train, H_train)
		
		return dt_classifier


	def naive_bayes(self, X_train, H_train):
		model = GaussianNB()
		model.fit(X_train, H_train)
		
		return model


	def linear_support_vector(self, X_train, H_train):
		svm_model = LinearSVC(random_state=0, max_iter=3500)
		svm_model.fit(X_train, H_train)
		
		return svm_model


	def get(self, request):
		exam, patient, med_hist = self.get_data()
		Features = [
			'age', 'sex', 'chest_pain', 'blood_systolic', 
			'blood_diastolic', 'chol_overall', 'smoke_per_day', 
			'smoker_years', 'fasting_glucose', 'hist_diabetes',
			'hist_heart_disease', 'heart_rate', 'exerc_angina'
		]

		dummies = ['sex', 'cp', 'fbs', 'dm', 'famhist', 'exang']
		columns_to_scale = ['age', 'trestbps', 'chol', 'cigs', 'years', 'thalrest', 'trestbpd']

		dummies2 = ['sex', 'chest_pain', 'fasting_glucose', 'hist_diabetes', 'hist_heart_disease', 'exerc_angina']
		columns_to_scale2 = ['age', 'blood_systolic', 'chol_overall', 'smoke_per_day', 'smoker_years', 'heart_rate', 'blood_diastolic']

		# # TODO
		# # Read csv file
		# current_dir =  os.path.abspath(os.path.dirname(__file__))
		# parent_dir = os.path.abspath(current_dir + "/../")
		# pathHeart = parent_dir + '/Data/new_cleveland.csv'
		# heart = pd.read_csv(pathHeart)
		# print(heart.head())
		# standardScaler = StandardScaler()
		# heart[columns_to_scale] = standardScaler.fit_transform(heart[columns_to_scale])
		# print(heart.head())
		# H = heart['target']
		# X = heart.drop(['target'], axis = 1)
		# X_train, X_test, H_train, H_test = train_test_split(X, H, test_size = 0.01, random_state = 0)

		# # KNN
		# knn_classifier = KNeighborsClassifier(n_neighbors = 3)
		# knn_classifier.fit(X_train, H_train)
		# test_pred = knn_classifier.predict(X_test)


		# Load dataframe
		heart = self.load_dataframe()
		# plot_diagrams(heart)

		# Use dummy columns for the categorical features
		standardScaler = StandardScaler()
		heart = self.scale_values(heart, standardScaler)

		# Split dataset
		H = heart['target']
		X = heart.drop(['target'], axis = 1)
		X_train, X_test, H_train, H_test = self.split_dataset(X, H)

		# KNN
		knn_classifier = self.KNN(X_train, H_train, X_test, H_test)

		# Decision Tree
		dt_classifier = self.decision_tree(X_train, H_train, X_test, H_test, X)

		# Naive Bayes
		nb_classifier = self.naive_bayes(X_train, H_train, X_test, H_test)

		# Linear Support Vector
		lsv_classifier = self.linear_support_vector(X_train, H_train, X_test, H_test)

		# # Predict heart disease
		# # Extract values
		# exam_param = []
		# exam_df = []
		# exam_predict = []
		# for f in range(len(Features)):
		# 	exam_param.append(Features[f])
		# 	exam_param.append(exam_values[len(exam_values)-1][Features[f]])
		# 	exam_predict.append(exam_values[len(exam_values)-1][Features[f]])
		# 	exam_df.append(exam_param)
		# 	exam_param = []

		# exam_df = dict(exam_df)
		# print(exam_df)
		# exam_df = pd.DataFrame(exam_df, columns=Features, index=[1])
		# print(exam_df)
		# # exam_df = pd.get_dummies(exam_df, columns = dummies2)
		# # standardScaler = StandardScaler()
		# exam_df[columns_to_scale2] = standardScaler.transform(exam_df[columns_to_scale2])
		# print(exam_df)
		# # Pass prediction 
		# Row_list =[]
		# # Iterate over each row 
		# for i in range((exam_df.shape[0])):
		# 	Row_list.append(list(exam_df.iloc[i, :])) 
		# # Print the list 
		# print(Row_list) 
		# prediction = knn_classifier.predict(Row_list)
		# print(prediction)




		return render(request, self.template_name, {'exams':exams, 'prediction': prediction})






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
