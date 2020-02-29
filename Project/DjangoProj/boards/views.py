from django.shortcuts import render, redirect
from django.urls import reverse
from urllib.parse import urlencode
from django.conf import settings
from django.views.generic import TemplateView
from boards.forms import ExamForm, CreatePatientForm, SelectPatientForm, FurtherActionsForm
from boards.models import Examination, Patient, Visit, Medical_history, Investigation

# For Machine learning model
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


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

			# Get patient and visit object linked to this exam
			p_name = request.GET.get('patient')
			patient = Patient.objects.get(patient_name=p_name)
			visit = Visit.objects.filter(patient=patient).latest('date')

			# Save data to  model
			exam = form.save(commit=False)
			exam.user = request.user
			exam.visit = visit
			exam.save()
			exam_input = form.cleaned_data

			# Pass the patient name to the results page
			p_arg = {}
			p_arg['patient'] = p_name
			base_url = reverse('results')
			query_string = urlencode(p_arg)
			url = '{}?{}'.format(base_url, query_string)
			return redirect(url)

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
	
	def get_data(self, request):
		p_name = request.GET.get('patient')

		patient = Patient.objects.filter(patient_name=p_name)
		# print(patient)
		visit = Visit.objects.filter(patient__in=patient).latest('date')
		# print(visit)
		# print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
		exams = Examination.objects.filter(visit=visit)
		# print(exams)
		med_hist = Medical_history.objects.filter(patient__in=patient)
		# print(med_hist)

		exam_vals = exams.values()[0]
		# print(exam_vals)
		patient_vals = patient.values_list('age', 'sex')[0]
		# print(patient_vals)
		med_hist_vals = med_hist.values(
			'heart_attack', 'angina', 'breathlessness',
			'chest_pain', 'high_chol', 'high_bp',
			'diabetes'
			)[0]
		# print(med_hist_vals)

		# Check if glucose is above 120 for heart dataset
		fbs = 0
		if (exam_vals['fasting_glucose'] > 120):
			fbs = 1
		
		# Extract heart data
		heart_vals = []
		heart_vals.extend(patient_vals)
		heart_vals.append(med_hist_vals['chest_pain'])
		heart_vals.append(exam_vals['blood_systolic'])
		heart_vals.append(exam_vals['blood_diastolic'])
		heart_vals.append(exam_vals['chol_overall'])
		heart_vals.append(exam_vals['smoke_per_day'])
		heart_vals.append(exam_vals['smoker_years'])
		heart_vals.append(fbs)
		heart_vals.append(0 if med_hist_vals['diabetes']==False else 1)
		heart_vals.append(0 if med_hist_vals['heart_attack']==False else 1)
		heart_vals.append(exam_vals['heart_rate'])
		heart_vals.append(0 if med_hist_vals['angina']==False else 1)
		print(heart_vals)


		# Extract diabetes data
		diabetes_vals = []
		diabetes_vals.append(0 if med_hist_vals['breathlessness']==False else 1)
		diabetes_vals.append(med_hist_vals['chest_pain'])
		diabetes_vals.append(0 if med_hist_vals['high_chol']==False else 1)
		diabetes_vals.append(0 if med_hist_vals['high_bp']==False else 1)
		
		# Calculate BMI
		height = exam_vals['height']
		weight = exam_vals['weight']
		bmi = weight / (height*height)
		diabetes_vals.append(bmi)

		diabetes_vals.append(0 if exam_vals['reg_pulse']==False else 1)
		diabetes_vals.append(exam_vals['pulse_type'])
		diabetes_vals.append(exam_vals['blood_systolic'])
		diabetes_vals.append(exam_vals['blood_diastolic'])
		diabetes_vals.append(exam_vals['protein'])
		diabetes_vals.append(exam_vals['hdl_chol'])
		diabetes_vals.append(exam_vals['ldl_chol'])
		diabetes_vals.append(exam_vals['chol_overall'])
		diabetes_vals.append(exam_vals['fasting_glucose'])
		diabetes_vals.append(exam_vals['triglyceride'])
		diabetes_vals.append(exam_vals['uric_acid'])

		return heart_vals, diabetes_vals, patient


	def load_heart(self):
		# Load heart disease dataset into pandas dataframe
		current_dir =  os.path.abspath(os.path.dirname(__file__))
		parent_dir = os.path.abspath(current_dir + "/../")
		pathHeart = parent_dir + '/Data/new_cleveland.csv'
		heart = pd.read_csv(pathHeart)

		return heart

	def load_diabetes(self):
		# Load heart disease dataset into pandas dataframe
		current_dir =  os.path.abspath(os.path.dirname(__file__))
		parent_dir = os.path.abspath(current_dir + "/../")
		pathDiabetes = parent_dir + '/Data/Diabetes.csv'
		diabetes = pd.read_csv(pathDiabetes)

		return diabetes


	def scale_heart(self, heart):
		heart = pd.get_dummies(heart, columns = ['cp'])
		columns_to_scale = ['age', 'trestbps', 'chol', 'cigs', 'years', 'thalrest', 'trestbpd']
		min_max_scaler = preprocessing.MinMaxScaler()
		heart[columns_to_scale] = min_max_scaler.fit_transform(heart[columns_to_scale])

		return heart, min_max_scaler, columns_to_scale

	def scale_diabetes(self, diabetes):
		columns_to_scale = ['BMI', 'Sys_BP', 'Dias_BP', 'Protein', 'HDL_Chol', 'LDL_Chol', 'Total_Chol', 'Fast_Glucose', 'Triglyceride', 'Uric_Acid']
		min_max_scaler = preprocessing.MinMaxScaler()
		diabetes[columns_to_scale] = min_max_scaler.fit_transform(diabetes[columns_to_scale])

		return diabetes, min_max_scaler, columns_to_scale


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
		heart_vals, diabetes_vals, patient = self.get_data(request)
		print(diabetes_vals)

		# Load dataframes
		heart = self.load_heart()

		# Put new data into dataframe
		heart_vals = pd.DataFrame(heart_vals).transpose()
		heart_vals.columns = heart.drop(['target'], axis=1).columns

		# Use dummy columns for the categorical features
		heart, min_max_scaler, columns_to_scale = self.scale_heart(heart)

		# Split dataset
		H = heart['target']
		X = heart.drop(['target'], axis = 1)
		X_train, X_test, H_train, H_test = self.split_dataset(X, H)

		# KNN
		knn_classifier = self.KNN(X_train, H_train)

		# Decision Tree
		dt_classifier = self.decision_tree(X_train, H_train)

		# Naive Bayes
		nb_classifier = self.naive_bayes(X_train, H_train)

		# Linear Support Vector
		lsv_classifier = self.linear_support_vector(X_train, H_train)
		
		# Scaling the new instance and getting dummies for cp col
		heart_vals = pd.get_dummies(heart_vals, columns = ['cp'])
		heart_vals[columns_to_scale] = min_max_scaler.transform(heart_vals[columns_to_scale])
		heart_vals = heart_vals.reindex(columns=X.columns, fill_value=0)

		# Making prediction
		heart_pred = knn_classifier.predict(heart_vals)


		# Diabetes
		# Load dataframes
		diabetes = self.load_diabetes()

		# Put new data into dataframe
		diabetes_vals = pd.DataFrame(diabetes_vals).transpose()
		diabetes_vals.columns = diabetes.drop(['Diabetes'],axis=1).columns

		# Use dummy columns for the categorical features
		diabetes, min_max_scaler, columns_to_scale = self.scale_diabetes(diabetes)
		
		# Split dataset
		D = diabetes['Diabetes']
		X = diabetes.drop(['Diabetes'], axis = 1)
		# X_train, X_test, D_train, D_test = self.split_dataset(X, D)

		# With oversampling
		sm = SMOTE(random_state=52)
		x_sm, d_sm = sm.fit_sample(X, D)
		X_train, X_test, D_train, D_test = self.split_dataset(x_sm, d_sm)

		# KNN
		knn_classifier = self.KNN(X_train, D_train)

		# Decision Tree
		dt_classifier = self.decision_tree(X_train, D_train)

		# Naive Bayes
		nb_classifier = self.naive_bayes(X_train, D_train)

		# Linear Support Vector
		lsv_classifier = self.linear_support_vector(X_train, D_train)
		
		# Scaling the new instance
		diabetes_vals[columns_to_scale] = min_max_scaler.transform(diabetes_vals[columns_to_scale])

		# Making prediction
		diabetes_pred = knn_classifier.predict(diabetes_vals)


		further_action_form = FurtherActionsForm(request.POST)
		if further_action_form.is_valid():
			# Save data to model
			visit = Visit.objects.filter(patient=patient).latest('date')
			further_action = further_action_form.save(commit=False)
			further_action.visit = request.user
			further_action.save()
			patient_input = further_action.cleaned_data
			return redirect('')


		# Send predictions
		predictions = {'heart_pred': heart_pred, 'diabetes_pred': diabetes_pred}
		return render(request, self.template_name, predictions)