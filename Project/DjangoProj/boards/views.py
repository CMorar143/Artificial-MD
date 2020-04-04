from django.shortcuts import render, redirect
from django.urls import reverse
from urllib.parse import urlencode
from django.views.generic import TemplateView
from boards.forms import ExamForm, CreatePatientForm, SelectPatientForm, FurtherActionsForm, CreateVisitForm
from boards.models import Examination, Patient, Patient_Ailment, Patient_Allergy, Patient_Medication, Visit, Medical_history, Investigation, Reminder, User, Ailment, Allergy, Medication
from django.contrib.auth.models import User, Group
from datetime import datetime
from dateutil.relativedelta import relativedelta
from django.db.models import Q
from django.contrib.auth.signals import user_logged_in
from django.dispatch import receiver

# For Machine learning model
import pandas as pd
import numpy as np
import os
# from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing
from sklearn.externals import joblib

first_login = False

@receiver(user_logged_in)
def logged_in(sender, **kwargs):
	if kwargs['user'].groups.filter(name='Receptionists').exists():
		global first_login
		first_login = True

user_logged_in.connect(logged_in)

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
			
			visit = Visit.objects.filter(patient=patient).filter(Q(date__lt=datetime.now()) | Q(date=datetime.now())).latest('date')
			med_hist = Medical_history.objects.filter(patient=patient).latest('date')

			# Save data to  model
			exam = Examform.save(commit=False)
			exam.user = request.user
			exam.visit = visit
			exam.save()
			exam_input = Examform.cleaned_data
			print(exam_input)
			
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
		# Check if this is a receptionist
		
		if request.GET.get('patient') is None:
			Createform = CreatePatientForm()
			Selectform = SelectPatientForm()

			args = {'Createform': Createform, 'Selectform': Selectform}			
			args = self.search_patients(request, args)

		else:
			args = self.display_info(request)

		if request.user.groups.filter(name='Doctors').exists():
			args['user_type'] = 'Doctors'
		
		elif request.user.groups.filter(name='Receptionists').exists():
			args['user_type'] = 'Receptionists'
			
			global first_login
			if first_login == True:
				overdue_rem = Reminder.objects.filter(Q(rem_date__lt=datetime.now()) | Q(rem_date=datetime.now())).order_by('rem_date')
				first_login = False
				
				args['first_login'] = True
				args['overdue_rem'] = overdue_rem

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
				print("IT ENTERED HERE \n\n\n\n\n")
				visit = Visitform.save(commit=False)
				visit.patient = patient
				visit.save()
			
			args = {'patient': patient, 'Visitform': Visitform, 'Selectform': Selectform}
			args = self.display_info(request)
			return render(request, self.template_name, args)

		elif 'start_exam' in request.POST:
			p_name = request.GET.get('patient')
			patient = Patient.objects.get(patient_name=p_name)
			visit = Visit.objects.filter(patient=patient)
			
			if visit:
				visit = visit.filter(Q(date__lt=datetime.now()) | Q(date=datetime.now().strftime("%Y-%m-%d %H:%M:00"))).latest('date')
			
			# Update the doctor if neccessary
			if visit.doctor != request.user and request.user.groups.filter(name='Doctors').exists():
				visit.doctor = request.user
				visit.save()

			p_arg = {}
			p_arg['patient'] = p_name
			base_url = reverse('exam')
			query_string = urlencode(p_arg)
			url = '{}?{}'.format(base_url, query_string)
			return redirect(url)
			


class reminders(TemplateView):
	template_name = 'reminders.html'

	def get(self, request):
		args = {}
		upcoming_rem = Reminder.objects.filter(rem_date__gt=datetime.now()).order_by('rem_date')
		
		overdue_rem = Reminder.objects.filter(Q(rem_date__lt=datetime.now()) | Q(rem_date=datetime.now())).order_by('rem_date')

		args['upcoming_rem'] = upcoming_rem
		args['overdue_rem'] = overdue_rem

		return render(request, self.template_name, args)



class diary(TemplateView):
	template_name = 'diary.html'

	def get(self, request):
		args = {}
		
		# Get only visits in the future to display
		visits = Visit.objects.filter(Q(date__gt=datetime.now()) | Q(date=datetime.now().strftime("%Y-%m-%d %H:%M:00"))).order_by('date')

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
		print(patient)
		visit = Visit.objects.filter(patient__in=patient)
		visit = visit.filter(Q(date__lt=datetime.now()) | Q(date=datetime.now())).latest('date')
		
		exams = Examination.objects.filter(visit=visit)
		# print(exams)
		med_hist_vals = Medical_history.objects.filter(patient__in=patient).latest('date')
		# print(med_hist['chest_pain'])

		exam_vals = exams.values().last()
		print(exam_vals)
		print("\n\n\n\n")
		
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
		# heart_vals.append(0 if med_hist_vals.diabetes==False else 1)
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
		heart = heart.drop(['dm'], axis=1)


		return heart

	def load_diabetes(self):
		# Load heart disease dataset into pandas dataframe
		current_dir =  os.path.abspath(os.path.dirname(__file__))
		parent_dir = os.path.abspath(current_dir + "/../")
		pathDiabetes = parent_dir + '/Data/Diabetes.csv'
		diabetes = pd.read_csv(pathDiabetes)

		return diabetes

	def load_dt(self, dt):
		current_dir =  os.path.abspath(os.path.dirname(__file__))
		parent_dir = os.path.abspath(current_dir + "/../")
		pathDT = parent_dir + '/Classifiers/' + dt

		decision_tree = joblib.load(pathDT)

		return decision_tree


	def bin_heart(self, heart, has_chest_pain):
		if has_chest_pain:
			heart = pd.get_dummies(heart, columns = ['cp'])
		columns_to_bin = ['age', 'trestbps', 'trestbpd', 'chol', 'cigs', 'years', 'thalrest']

		for col in columns_to_bin:
			# Chol requires more buckets
			if col == 'chol':
				heart[col] = pd.cut(heart[col], 10)
			else:
				heart[col] = pd.cut(heart[col], 7)

		# heart = pd.get_dummies(heart, columns = columns_to_bin)

		return heart

	def bin_diabetes(self, diabetes):
		columns_to_bin = ['BMI', 'Sys_BP', 'Dias_BP', 'HDL_Chol', 'LDL_Chol', 
						'Total_Chol', 'Fast_Glucose', 'Triglyceride', 'Uric_Acid']

		for col in columns_to_bin:
			if col == 'Uric_Acid':
				diabetes[col] = pd.cut(diabetes[col], 2)
			else:
				diabetes[col] = pd.cut(diabetes[col], 8)

		# diabetes = pd.get_dummies(diabetes, columns = columns_to_bin)

		return diabetes

	def get_target_entropy(self, df):
		entropy = 0

		# Possible values are they have heart disease or they don't (1 or 0 respectively)
		values = df['target'].unique()

		# Calculate entropy
		for value in values:
			val_split = df['target'].value_counts()[value]/len(df['target'])
			entropy = entropy + -val_split*np.log2(val_split)

		return entropy

	def get_feature_entropy(self, df, feature):
		feature_entropy = 0

		# To prevent the feature entropies from being null
		smallest_num = np.finfo(float).tiny

		# Get the unique values for the target and the feature
		values = df['target'].unique()
		feature_vals = df[feature].unique()

		for value in feature_vals:
			val_entropy = 0
			for val in values:
				# Get the number of possible values within the feature
				num_of_each_val = df[feature][df[feature]==value]
				
				# For getting the ratio
				numerator = len(num_of_each_val[df['target']==val])
				denominator = len(num_of_each_val)
				
				# Add the smallest number so its not dividing by 0
				val_split = numerator/(denominator+smallest_num)
				
				""" Get the entropy for both target feature 
					values with respect to this feature value
				"""
				# Add the smallest number so its not log2(0)
				val_entropy = val_entropy + -val_split*np.log2(val_split+smallest_num)

			# Get the entropy for all values in this feature
			val_ratio = denominator/len(df)
			feature_entropy = feature_entropy + val_ratio*val_entropy
		
		return feature_entropy

	def calc_info_gains(self, df, info_gains):
		# Calculate the info_gain for non-target features only
		features = df.drop(['target'], axis=1)

		# Get entropy of target feature
		target_entropy = self.get_target_entropy(df)
		# print(target_entropy)
		# print(features)
		for f in features:
			feature_entropy = self.get_feature_entropy(df, f)
			information_gain = target_entropy - feature_entropy
			info_gains[f] = information_gain

		return info_gains

	def find_feature(self, df, info_gains):
		info_gains = self.calc_info_gains(df, info_gains)
		# print(info_gains)
		vals = list(info_gains.values())
		feat = list(info_gains.keys())

		return feat[vals.index(max(vals))]

	def create_tree(self, df, dec_tree = 0):
		# Find the feature to split on i.e. the node feature
		info_gains = {}
		node_feature = self.find_feature(df, info_gains)
		# print(node_feature)
		node_feat_vals = df[node_feature]

		# Initialise decision tree
		if dec_tree == 0:
			dec_tree = {}
			dec_tree[node_feature] = {}

		# Get all values for the node
		all_node_vals = np.unique(node_feat_vals)
		print(all_node_vals)
		print(node_feature)
		# Build the tree with recursion
		for val in all_node_vals:
			sub_tree = df[node_feat_vals == val].reset_index(drop=True)

			values, size = np.unique(sub_tree['target'], return_counts=True)
			print(val)
			print(values)
			print(len(size))
			# More of the tree needs to be built
			if len(size) > 1:
				without_target = sub_tree.drop(['target'], axis=1)
				no_duplicates = without_target.drop_duplicates(without_target.columns)
				
				if len(no_duplicates) == 1:
					print("THEY'RE EQUAL\n\n\n")
					continue
				else:
					print("Making recursive call\n\n\n")
					dec_tree[node_feature][val] = self.create_tree(sub_tree) 
			
			# This is the leaf node
			else:
				dec_tree[node_feature][val] = values[0]

		return dec_tree

	def make_prediction(self, new_data, decision_tree):
		# Start at the root node
		root = list(decision_tree.keys())

		# Loop through all possible sub nodes
		for sub_node in root:
			
			# Getting the value of the root node for the new data point
			val = new_data[sub_node]
			print(decision_tree.keys())
			# Getting the subtree at that value
			decision_tree = decision_tree[sub_node][val]
			pred = 0

			# If the subtree has its own subtree then make the recursive call
			if type(decision_tree) == type({}):
				pred = self.make_prediction(new_data, decision_tree)
			
			# The subtree just contains the prediction
			else:
				pred = decision_tree

		return pred


	def get(self, request):
		heart_vals, diabetes_vals, has_chest_pain = self.get_data(request)

		# Load dataframes
		heart = self.load_heart()
		data = np.array([29,1,1,131,87,205,5,4,0,0,75,0])
		heart_vals = pd.Series(data, index=['age','sex','cp','trestbps','trestbpd',
										'chol','cigs','years','fbs','famhist','thalrest',
										'exang'])
		# Put new data into dataframe
		# heart_vals = pd.DataFrame(heart_vals).transpose()
		
		if has_chest_pain == False:
			heart = heart.drop(['cp'], axis = 1)

		heart_vals.columns = heart.drop(['target'], axis=1).columns

		heart = self.bin_heart(heart, has_chest_pain)

		# Check if a new tree needs to be created
		need_updated_tree = False
		columns_to_check = ['age', 'trestbps', 'trestbpd', 'chol', 'thalrest']
		
		for col in columns_to_check:
			check = [heart_vals[col] in x for x in heart[col].unique()]
			if True not in check:
				need_updated_tree = True
		
		heart = self.load_heart()
		heart = heart.append(heart_vals, ignore_index=True)
		heart = self.bin_heart(heart, has_chest_pain)

		heart_vals = heart.drop(['target'], axis=1).iloc[-1]
		heart = heart.drop(heart.index[-1])

		# Build tree
		if need_updated_tree:
			heart_dt = self.create_tree(heart)

		else:
			if has_chest_pain:
				heart_dt = self.load_dt('heart_dt_hascp.pkl')
			else:
				heart_dt = self.load_dt('heart_dt.pkl')

		heart_pred = 0
		# Make predictions
		heart_pred = self.make_prediction(heart_vals, heart_dt)
		
		print("\n\n\n")

		# Diabetes
		# Load dataframes
		diabetes = self.load_diabetes()
		data = np.array([1,1,0,0,28.9,0,140,90,60,56,126,193,51,5.7])
		diabetes_vals = pd.Series(data, index=['Short_Breath', 'Chest_Pains', 'High_Chol_Hist',
									'High_BP_Hist',	'BMI', 'Reg_Pulse',	'Sys_BP', 'Dias_BP',
									'HDL_Chol', 'LDL_Chol', 'Total_Chol', 'Fast_Glucose', 'Triglyceride',
									'Uric_Acid'])
		
		# In order to use the methods to create the tree
		# diabetes.rename(columns={'Diabetes': 'target'}, inplace=True)
		print(diabetes.columns)

		# Put new data into dataframe
		# diabetes_vals = pd.DataFrame(diabetes_vals).transpose()
		diabetes_vals.columns = diabetes.drop(['Diabetes'], axis=1).columns

		diabetes = self.bin_diabetes(diabetes)

		need_updated_tree = False
		columns_to_check = ['BMI', 'Sys_BP', 'Dias_BP', 'HDL_Chol', 'LDL_Chol', 'Total_Chol', 'Fast_Glucose', 'Triglyceride', 'Uric_Acid']
		
		for col in columns_to_check:
			check = [diabetes_vals[col] in x for x in diabetes[col].unique()]
			print(check)
			if True not in check:
				need_updated_tree = True

		diabetes = self.load_diabetes()
		diabetes = diabetes.append(diabetes_vals, ignore_index=True)
		diabetes = self.bin_diabetes(diabetes)

		diabetes_vals = diabetes.drop(['Diabetes'], axis=1).iloc[-1]
		diabetes = diabetes.drop(diabetes.index[-1])

		# Build tree
		if need_updated_tree:
			# In order to use the methods to create the tree
			diabetes.rename(columns={'Diabetes': 'target'}, inplace=True)
			diabetes_dt = self.create_tree(diabetes)

		else:
			diabetes_dt = self.load_dt('diabetes_dt.pkl')

		diabetes_pred = 0
		# Make predictions
		diabetes_pred = self.make_prediction(diabetes_vals, diabetes_dt)

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

