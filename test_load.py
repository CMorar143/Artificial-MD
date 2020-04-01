import pandas as pd
import numpy as np
import pickle
from sklearn.externals import joblib

def load_dataframe():
	# Load diabetes disease dataset into pandas dataframe
	pathdiabetes = "../../FYP_Data/Health_Survey/"
	diabetes = pd.read_csv(pathdiabetes + 'Diabetes.csv')

	return diabetes

def bin_values(diabetes):
	columns_to_bin = ['BMI', 'Sys_BP', 'Dias_BP', 'HDL_Chol', 'LDL_Chol', 'Total_Chol', 'Fast_Glucose', 'Triglyceride', 'Uric_Acid']
	
	for col in columns_to_bin:
		if col == 'Uric_Acid':
			diabetes[col] = pd.cut(diabetes[col], 2)
		else:
			diabetes[col] = pd.cut(diabetes[col], 8)
	
	# diabetes = pd.get_dummies(diabetes, columns = columns_to_bin)

	return diabetes


def make_prediction(new_data, decision_tree):
	# Start at the root node
	root = list(decision_tree.keys())

	# Loop through all possible sub nodes
	for sub_node in root:
		
		# Getting the value of the root node for the new data point
		val = new_data[sub_node]
		
		# Getting the subtree at that value
		decision_tree = decision_tree[sub_node][val]
		pred = 0

		# If the subtree has its own subtree then make the recursive call
		if type(decision_tree) == type({}):
			pred = make_prediction(new_data, decision_tree)
		
		# The subtree just contains the prediction
		else:
			pred = decision_tree

	return pred


def main():
	# Load dataset
	diabetes = load_dataframe()
	data = np.array([21,1,1,131,87,205,5,4,0,0,75,0])
	instance = pd.Series(data, index=['age','sex','cp','trestbps','trestbpd',
									'chol','cigs','years','fbs','famhist','thalrest',
									'exang'])

	# print(instance)

	# diabetes = diabetes.append(instance, ignore_index=True)

	# Bin features
	diabetes = bin_values(diabetes)

	print(diabetes.tail())

	new_data = diabetes.drop(['Diabetes'], axis=1).iloc[-1]
	diabetes = diabetes.drop(diabetes.index[-1])

	print(diabetes.tail())
	# print(new_data)
	# Build tree
	decision_tree = joblib.load('diabetes_dt.pkl')
	
	# print(diabetes.iloc[4])

	# new_data = diabetes.drop(['Diabetes'], axis=1).iloc[4]

	# Make predictions
	pred = make_prediction(new_data, decision_tree)

	print(pred)

main()