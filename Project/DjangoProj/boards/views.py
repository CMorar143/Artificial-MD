from django.shortcuts import render, redirect
from django.urls import reverse
from urllib.parse import urlencode
# from django.conf import settings
from django.views.generic import TemplateView
from boards.forms import ExamForm, CreatePatientForm, SelectPatientForm, FurtherActionsForm, CreateVisitForm
from boards.models import Examination, Patient, Patient_Ailment, Patient_Allergy, Patient_Medication, Visit, Medical_history, Investigation, Reminder, User, Ailment, Allergy, Medication
from datetime import datetime
from dateutil.relativedelta import relativedelta
from django.db.models import Q

# For Machine learning model
import pandas as pd
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


class home(TemplateView):
	template_name = 'home.html'

	def get(self, request):
		if request.user.is_authenticated:
			return redirect('logout')
		else:
			return redirect('login')
		# return render(request, self.template_name)


class exam(TemplateView):
	template_name = 'exam.html'

	def get(self, request):
		Examform = ExamForm()
		return render(request, self.template_name, {'Examform': Examform})

	def post(self, request):
		Examform = ExamForm(request.POST)
		if Examform.is_valid():

			# Get patient and visit object linked to this exam
			p_name = request.GET.get('patient')
			patient = Patient.objects.get(patient_name=p_name)
			# visit = Visit.objects.filter(patient=patient).latest('date')
			visit = Visit.objects.filter(patient=patient).filter(Q(date__lt=datetime.now()) | Q(date=datetime.now())).latest('date')
			med_hist = Medical_history.objects.filter(patient=patient).latest('date')
			
			# Save data to  model
			exam = Examform.save(commit=False)
			exam.user = request.user
			exam.visit = visit
			exam.save()
			exam_input = Examform.cleaned_data
			
			# Check if their hist is changed, if it has, insert that into the medical_history model
			if int(exam_input.get('cp')) != med_hist.chest_pain and bool(exam_input.get('breathlessness')) != med_hist.breathlessness:
				m_hist = Medical_history(patient=patient, chest_pain=int(exam_input.get('cp')), breathlessness=bool(exam_input.get('breathlessness')))
				m_hist.save()
			elif int(exam_input.get('cp')) != med_hist.chest_pain:
				m_hist = Medical_history(patient=patient, chest_pain=int(exam_input.get('cp')))
				m_hist.save()
			elif bool(exam_input.get('breathlessness')) != med_hist.breathlessness:
				m_hist = Medical_history(patient=patient, breathlessness=bool(exam_input.get('breathlessness')))
				m_hist.save()

			# Pass the patient name to the results page
			p_arg = {}
			p_arg['patient'] = p_name
			base_url = reverse('results')
			query_string = urlencode(p_arg)
			url = '{}?{}'.format(base_url, query_string)
			return redirect(url)

		args = {'Examform': Examform}
		return render(request, self.template_name, args)


class patient_info(TemplateView):

	def get(self, request, id):
		# Get information for patient page
		patient = Patient.objects.get(id=id)					

		# Pass the patient name to the patient page
		p_arg = {}
		p_arg['patient'] = patient.patient_name
		base_url = reverse('patient')
		query_string = urlencode(p_arg)
		url = '{}?{}'.format(base_url, query_string)
		return redirect(url)


class patient(TemplateView):
	template_name = 'patient.html'
	patient_info = 'patient_info.html'

	def search_patients(self, request, args):
		if 'search' in request.GET:
			searched_name = request.GET['search']
			searched_patients = Patient.objects.filter(patient_name__icontains=searched_name)
			if searched_patients is not None:
				args['searched_patients'] = searched_patients
		
		return args

	def display_info(self, request):
		Selectform = SelectPatientForm()
		Visitform = CreateVisitForm()

		# Get information for patient page
		p_name = request.GET.get('patient')
		patient = Patient.objects.get(patient_name=p_name)

		reminders = Reminder.objects.filter(patient_id=patient.id).order_by('rem_date')				
		
		allergy_ids = Patient_Allergy.objects.filter(patient_id=patient.id).values('allergy_id')
		allergies = Allergy.objects.filter(id__in=allergy_ids)

		ailment_ids = Patient_Ailment.objects.filter(patient_id=patient.id).values('ailment_id')
		ailments = Ailment.objects.filter(id__in=ailment_ids)
		
		med_ids = Patient_Medication.objects.filter(patient_id=patient.id).values('medication_id')
		medication = Medication.objects.filter(id__in=med_ids)

		args = {'patient': patient, 'Visitform': Visitform, 'Selectform': Selectform}
		args = self.search_patients(request, args)

		if reminders.exists():
			args['reminders'] = reminders

		if allergies.exists():
			args['allergies'] = allergies

		if ailments.exists():
			args['ailments'] = ailments

		if medication.exists():
			args['medication'] = medication

		return args

	def get(self, request):
		if request.GET.get('patient') is None:
			Createform = CreatePatientForm()
			Selectform = SelectPatientForm()

			searched_name = ''
			args = {'Createform': Createform, 'Selectform': Selectform}			
			args = self.search_patients(request, args)

		else:
			args = self.display_info(request)

		return render(request, self.template_name, args)

	def post(self, request):
		if 'create_patient' in request.POST:
			Createform = CreatePatientForm(request.POST)
			Selectform = SelectPatientForm()
			if Createform.is_valid():

				# Save data to model
				patient = Createform.save(commit=False)
				patient.user = request.user
				patient.save()
				patient_input = Createform.cleaned_data

				# Pass the patient name to the patient page
				p_arg = {}
				p_arg['patient'] = patient.patient_name
				base_url = reverse('patient')
				query_string = urlencode(p_arg)
				url = '{}?{}'.format(base_url, query_string)
				return redirect(url)

			args = {'Createform': Createform, 'Selectform': Selectform}
			return render(request, self.template_name, args)

		elif 'create_visit' in request.POST:
			Visitform = CreateVisitForm(request.POST)
			Selectform = SelectPatientForm()
			p_name = request.GET.get('patient')
			patient = Patient.objects.get(patient_name=p_name)
				
			if Visitform.is_valid():
				
				# Save data to model
				visit = Visitform.save(commit=False)
				visit.doctor = request.user
				visit.patient = patient
				visit.save()

				if visit.reason in 'Examination' and str(visit.date) in datetime.now().strftime("%Y-%m-%d %H:%M:00"):
					p_arg = {}
					p_arg['patient'] = p_name
					base_url = reverse('exam')
					query_string = urlencode(p_arg)
					url = '{}?{}'.format(base_url, query_string)
					return redirect(url)
			
			args = {'patient': patient, 'Visitform': Visitform, 'Selectform': Selectform}
			args = self.display_info(request)
			return render(request, self.template_name, args)

		# elif 'start_exam' in request.POST:
		# 	p_name = request.GET.get('patient')
		# 	patient = Patient.objects.get(patient_name=p_name)
		# 	visit = Visit.objects.filter(patient=patient)
			
		# 	if visit:
		# 		visit = visit.latest('date')
		# 		# If they dont have a visit for today, create one
			
		# 	if not visit or visit.date.date() != date.today():
		# 		v = Visit(reason="Examination", doctor=request.user, patient=patient)
		# 		v.save()
			
		# 	p_arg = {}
		# 	p_arg['patient'] = p_name
		# 	base_url = reverse('exam')
		# 	query_string = urlencode(p_arg)
		# 	url = '{}?{}'.format(base_url, query_string)
		# 	return redirect(url)



class reminders(TemplateView):
	template_name = 'reminders.html'

	def get(self, request):
		args = {}
		upcoming_rem = Reminder.objects.filter(rem_date__gt=datetime.now()).order_by('rem_date')
		
		# Change to less than OR equal to today
		overdue_rem = Reminder.objects.filter(Q(rem_date__lt=datetime.now()) | Q(rem_date=datetime.now())).order_by('rem_date')

		args['upcoming_rem'] = upcoming_rem
		args['overdue_rem'] = overdue_rem
		print(upcoming_rem)
		print(overdue_rem)
		return render(request, self.template_name, args)



class diary(TemplateView):
	template_name = 'diary.html'

	def get(self, request):
		args = {}
		
		# Get only visits in the future to display
		visits = Visit.objects.filter(date__gt=datetime.now()).order_by('date')

		args['visits'] = visits

		return render(request, self.template_name, args)



class results(TemplateView):
	template_name = 'results.html'

	def get_age(self, patient):
		# Get current time
		current_date = datetime.date(datetime.now())
		
		# Get birthdate
		birth_date = patient.values_list('DOB')[0][0]
		
		# Get difference in years
		time_difference = relativedelta(current_date, birth_date)
		age = time_difference.years
		
		return age
	
	def get_data(self, request):
		has_chest_pain = False

		p_name = request.GET.get('patient')
		patient = Patient.objects.filter(patient_name=p_name)
		# print(patient)
		visit = Visit.objects.filter(Q(date__lt=datetime.now()) | Q(date=datetime.now())).latest('date')
		# visit = Visit.objects.filter(patient__in=patient).latest('date')
		# print(visit)
		# print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
		exams = Examination.objects.filter(visit=visit)
		# print(exams)
		med_hist_vals = Medical_history.objects.filter(patient__in=patient).latest('date')
		# print(med_hist['chest_pain'])

		exam_vals = exams.values()[0]
		# print(exam_vals)
		
		# Get age and sex
		patient_vals = patient.values_list('sex')[0]
		age = self.get_age(patient)
		patient_vals = (age,) + patient_vals

		if med_hist_vals.chest_pain != 0:
			has_chest_pain = True

		# print(med_hist_vals)

		# Check if glucose is above 120 for heart dataset
		fbs = 0
		if (exam_vals['fasting_glucose'] > 120):
			fbs = 1
		
		# Extract heart data
		heart_vals = []
		heart_vals.extend(patient_vals)

		if has_chest_pain:
			print("entered has_chest_pain")
			heart_vals.append(med_hist_vals.chest_pain)
		heart_vals.append(exam_vals['blood_systolic'])
		heart_vals.append(exam_vals['blood_diastolic'])
		heart_vals.append(exam_vals['chol_total'])
		heart_vals.append(exam_vals['smoke_per_day'])
		heart_vals.append(exam_vals['smoker_years'])
		heart_vals.append(fbs)
		heart_vals.append(0 if med_hist_vals.diabetes==False else 1)
		heart_vals.append(0 if med_hist_vals.heart_attack==False else 1)
		heart_vals.append(exam_vals['heart_rate'])
		heart_vals.append(0 if med_hist_vals.angina==False else 1)
		print(heart_vals)


		# Extract diabetes data
		diabetes_vals = []
		diabetes_vals.append(0 if med_hist_vals.breathlessness==False else 1)
		diabetes_vals.append(0 if med_hist_vals.chest_pain==0 else 1)
		diabetes_vals.append(0 if med_hist_vals.high_chol==False else 1)
		diabetes_vals.append(0 if med_hist_vals.high_bp==False else 1)
		
		# Calculate BMI
		height = exam_vals['height']
		weight = exam_vals['weight']
		bmi = weight / (height*height)
		diabetes_vals.append(bmi)

		diabetes_vals.append(0 if exam_vals['reg_pulse']==False else 1)
		diabetes_vals.append(exam_vals['blood_systolic'])
		diabetes_vals.append(exam_vals['blood_diastolic'])
		diabetes_vals.append(exam_vals['hdl_chol'])
		diabetes_vals.append(exam_vals['ldl_chol'])
		diabetes_vals.append(exam_vals['chol_total'])
		diabetes_vals.append(exam_vals['fasting_glucose'])
		diabetes_vals.append(exam_vals['triglyceride'])
		diabetes_vals.append(exam_vals['uric_acid'])

		return heart_vals, diabetes_vals, has_chest_pain


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


	def scale_heart(self, heart, has_chest_pain):
		if has_chest_pain:
			heart = pd.get_dummies(heart, columns = ['cp'])
		columns_to_scale = ['age', 'trestbps', 'chol', 'cigs', 'years', 'thalrest', 'trestbpd']
		min_max_scaler = preprocessing.MinMaxScaler()
		heart[columns_to_scale] = min_max_scaler.fit_transform(heart[columns_to_scale])

		return heart, min_max_scaler, columns_to_scale

	def scale_diabetes(self, diabetes):
		columns_to_scale = ['BMI', 'Sys_BP', 'Dias_BP', 'HDL_Chol', 'LDL_Chol', 'Total_Chol', 'Fast_Glucose', 'Triglyceride', 'Uric_Acid']
		min_max_scaler = preprocessing.MinMaxScaler()
		diabetes[columns_to_scale] = min_max_scaler.fit_transform(diabetes[columns_to_scale])

		return diabetes, min_max_scaler, columns_to_scale


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
		heart_vals, diabetes_vals, has_chest_pain = self.get_data(request)

		# Load dataframes
		heart = self.load_heart()

		# Put new data into dataframe
		heart_vals = pd.DataFrame(heart_vals).transpose()

		if has_chest_pain == False:
			heart_vals.columns = heart.drop(['target', 'cp'], axis=1).columns
		else:
			heart_vals.columns = heart.drop(['target'], axis=1).columns

		# Use dummy columns for the categorical features
		heart, min_max_scaler, columns_to_scale = self.scale_heart(heart, has_chest_pain)

		if has_chest_pain == False:
			heart = heart.drop(['cp'], axis = 1)

		# Split dataset
		H = heart['target']
		X = heart.drop(['target'], axis = 1)

		# With oversampling
		sm = SMOTE(random_state=52)
		X, H = sm.fit_sample(X, H)
		
		# KNN
		knn_classifier = self.KNN(X, H)

		# Decision Tree
		dt_classifier = self.decision_tree(X, H)

		# Naive Bayes
		nb_classifier = self.naive_bayes(X, H)

		# Linear Support Vector
		lsv_classifier = self.linear_support_vector(X, H)
		
		# Scaling the new instance and getting dummies for cp col
		if has_chest_pain:
			heart_vals = pd.get_dummies(heart_vals, columns = ['cp'])
		heart_vals[columns_to_scale] = min_max_scaler.transform(heart_vals[columns_to_scale])
		heart_vals = heart_vals.reindex(columns=X.columns, fill_value=0)

		# Making prediction
		heart_pred = knn_classifier.predict(heart_vals)
		print(knn_classifier.predict(heart_vals))
		print(dt_classifier.predict(heart_vals))
		print(nb_classifier.predict(heart_vals))
		print(lsv_classifier.predict(heart_vals))
		print("\n\n\n")

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
		
		# With oversampling
		# sm = SMOTE(random_state=52)
		# X, D = sm.fit_sample(X, D)
		
		# KNN
		knn_classifier = self.KNN(X, D)

		# Decision Tree
		dt_classifier = self.decision_tree(X, D)

		# Naive Bayes
		nb_classifier = self.naive_bayes(X, D)

		# Linear Support Vector
		lsv_classifier = self.linear_support_vector(X, D)
		
		# Scaling the new instance
		diabetes_vals[columns_to_scale] = min_max_scaler.transform(diabetes_vals[columns_to_scale])

		# Making prediction
		diabetes_pred = knn_classifier.predict(diabetes_vals)
		print(knn_classifier.predict(diabetes_vals))
		print(dt_classifier.predict(diabetes_vals))
		print(nb_classifier.predict(diabetes_vals))
		print(lsv_classifier.predict(diabetes_vals))

		# Allow physician to choose further action
		further_action_form = FurtherActionsForm()

		# Send predictions
		args = {'heart_pred': heart_pred, 'diabetes_pred': diabetes_pred, 'further_action_form': further_action_form}
		return render(request, self.template_name, args)

	def post(self, request):
		p_name = request.GET.get('patient')
		patient = Patient.objects.filter(patient_name=p_name)
		
		further_action_form = FurtherActionsForm(request.POST)
		if further_action_form.is_valid():	
			# Save data to model
			further_action = further_action_form.save(commit=False)

			if further_action.further_actions != 'None':
				# Update the outcome of the visit
				visit = Visit.objects.filter(patient=patient).filter(Q(date__lt=datetime.now()) | Q(date=datetime.now())).latest('date')
				visit.outcome = further_action.further_actions
				visit.save()
				
				further_action.visit = visit
				
				# if further_action.further_actions in 'Follow up appointment':
					

				if further_action.further_actions != 'Referral':
					further_action.ref_to = None;
					further_action.ref_reason = None;
				# else:
					# reminder = Reminder(location=further_action.ref_to, message=further_action.ref_reason, patient=patient)
					# reminder.save()
				further_action.save()
				patient_input = further_action_form.cleaned_data
			# Send notification to the secretary
			# user = User.objects.get(username='PMorar')
			# print(user.first_name)

			# notify.send(request.user, recipient=user, actor=request.user,
			# verb='followed you.', nf_type='followed_by_one_user')

			p_arg = {}
			p_arg['patient'] = p_name
			base_url = reverse('patient')
			query_string = urlencode(p_arg)
			url = '{}?{}'.format(base_url, query_string)
			return redirect(url)

