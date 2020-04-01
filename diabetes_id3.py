import pandas as pd
import numpy as np
from sklearn.externals import joblib

def load_dataframe():
	# Load diabetes disease dataset into pandas dataframe
	pathdiabetes = "../../FYP_Data/Health_Survey/"
	diabetes = pd.read_csv(pathdiabetes + 'Diabetes.csv')

	return diabetes

def bin_values(diabetes):
	columns_to_bin = ['BMI', 'Sys_BP', 'Dias_BP', 'HDL_Chol', 'LDL_Chol', 
					'Total_Chol', 'Fast_Glucose', 'Triglyceride', 'Uric_Acid']
	
	for col in columns_to_bin:
		if col == 'Uric_Acid':
			diabetes[col] = pd.cut(diabetes[col], 2)
		else:
			diabetes[col] = pd.cut(diabetes[col], 8)
	
	# diabetes = pd.get_dummies(diabetes, columns = columns_to_bin)

	return diabetes

def get_target_entropy(diabetes):
	entropy = 0

	# Possible values are they have diabetes disease or they don't (1 or 0 respectively)
	values = diabetes['Diabetes'].unique()

	# Calculate entropy
	for value in values:
		val_split = diabetes['Diabetes'].value_counts()[value]/len(diabetes['Diabetes'])
		entropy = entropy + -val_split*np.log2(val_split)

	return entropy

def get_feature_entropy(diabetes, feature):
	feature_entropy = 0

	# To prevent the feature entropies from being null
	smallest_num = np.finfo(float).tiny

	# Get the unique values for the target and the feature
	values = diabetes['Diabetes'].unique()
	feature_vals = diabetes[feature].unique()

	for value in feature_vals:
		val_entropy = 0
		for val in values:
			# Get the number of possible values within the feature
			num_of_each_val = diabetes[feature][diabetes[feature]==value]
			
			# For getting the ratio
			numerator = len(num_of_each_val[diabetes['Diabetes']==val])
			denominator = len(num_of_each_val)
			
			# Add the smallest number so its not dividing by 0
			val_split = numerator/(denominator+smallest_num)
			
			""" Get the entropy for both target feature 
				values with respect to this feature value
			"""
			# Add the smallest number so its not log2(0)
			val_entropy = val_entropy + -val_split*np.log2(val_split+smallest_num)

		# Get the entropy for all values in this feature
		val_ratio = denominator/len(diabetes)
		feature_entropy = feature_entropy + val_ratio*val_entropy
	
	return feature_entropy

def calc_info_gains(diabetes, info_gains):
	# Calculate the info_gain for non-target features only
	features = diabetes.drop(['Diabetes'], axis=1)

	# Get entropy of target feature
	target_entropy = get_target_entropy(diabetes)

	for f in features:
		feature_entropy = get_feature_entropy(diabetes, f)
		information_gain = target_entropy - feature_entropy
		info_gains[f] = information_gain

	return info_gains

def find_feature(diabetes, info_gains):
	info_gains = calc_info_gains(diabetes, info_gains)

	vals = list(info_gains.values())
	feat = list(info_gains.keys())

	return feat[vals.index(max(vals))]

def create_tree(diabetes, dec_tree = 0):
	# Find the feature to split on i.e. the node feature
	info_gains = {}
	node_feature = find_feature(diabetes, info_gains)
	node_feat_vals = diabetes[node_feature]

	# Initialise decision tree
	if dec_tree == 0:
		dec_tree = {}
		dec_tree[node_feature] = {}

	# Get all values for the node
	all_node_vals = np.unique(node_feat_vals)
	print(node_feature)
	# Build the tree with recursion
	for val in all_node_vals:
		sub_tree = diabetes[node_feat_vals == val].reset_index(drop=True)

		values, size = np.unique(sub_tree['Diabetes'], return_counts=True)
		print(val)
		print(values)
		print(size)
		# More of the tree needs to be built
		if len(size) > 1:
			# print(dec_tree[node_feature])
			without_target = sub_tree.drop(['Diabetes'], axis=1)
			no_duplicates = without_target.drop_duplicates(without_target.columns)
			print(sub_tree.transpose())
			print(len(size))
			print(size[1])
			print(len(no_duplicates))
			# if node_feature == 'Short_Breath':
			if len(no_duplicates) == 1:
				print("THEY'RE EQUAL\n\n\n")
				continue
			else:
				print("Making recursive call\n\n\n")
				dec_tree[node_feature][val] = create_tree(sub_tree) 
		
		# This is the leaf node
		else:
			dec_tree[node_feature][val] = values[0]
			
	return dec_tree


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
	# data = np.array([21,1,1,131,87,205,5,4,0,0,75,0])
	# instance = pd.Series(data, index=['age','sex','cp','trestbps','trestbpd',
	# 								'chol','cigs','years','fbs','famhist','thalrest',
	# 								'exang'])

	# print(instance)

	# diabetes = diabetes.append(instance, ignore_index=True)

	# Bin features
	diabetes = bin_values(diabetes)

	print(diabetes.tail())

	new_data = diabetes.drop(['Diabetes'], axis=1).iloc[-1]
	# diabetes = diabetes.drop(diabetes.index[-1])

	print(diabetes.tail())
	# print(new_data)
	# Build tree
	decision_tree = create_tree(diabetes)
	joblib.dump(decision_tree, 'diabetes_dt.pkl')
	# print(diabetes.iloc[4])

	# new_data = diabetes.drop(['Diabetes'], axis=1).iloc[4]

	# Make predictions
	pred = make_prediction(new_data, decision_tree)

	print(pred)

main()