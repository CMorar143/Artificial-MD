import pandas as pd
import numpy as np
from sklearn.externals import joblib

def load_dataframe():
	# Load heart disease dataset into pandas dataframe
	pathHeart = "../../FYP_Data/heart-disease-uci/"
	heart = pd.read_csv(pathHeart + 'new_cleveland.csv')
	heart = heart.drop(['dm'], axis=1)

	return heart

def bin_values(heart):
	columns_to_bin = ['age', 'trestbps', 'trestbpd', 'chol', 'cigs', 'years', 'thalrest']

	for col in columns_to_bin:
		# Chol requires more buckets
		if col == 'chol':
			heart[col] = pd.cut(heart[col], 10)
		else:
			heart[col] = pd.cut(heart[col], 7)
	
	heart = pd.get_dummies(heart, columns = columns_to_bin)

	return heart, columns_to_bin

def get_target_entropy(heart):
	entropy = 0

	# Possible values are they have heart disease or they don't (1 or 0 respectively)
	values = heart['target'].unique()

	# Calculate entropy
	for value in values:
		val_split = heart['target'].value_counts()[value]/len(heart['target'])
		entropy = entropy + -val_split*np.log2(val_split)

	return entropy

def get_feature_entropy(heart, feature):
	feature_entropy = 0

	# To prevent the feature entropies from being null
	smallest_num = np.finfo(float).tiny

	# Get the unique values for the target and the feature
	values = heart['target'].unique()
	feature_vals = heart[feature].unique()

	for value in feature_vals:
		val_entropy = 0
		for val in values:
			# Get the number of possible values within the feature
			num_of_each_val = heart[feature][heart[feature]==value]
			
			# For getting the ratio
			numerator = len(num_of_each_val[heart['target']==val])
			denominator = len(num_of_each_val)
			
			# Add the smallest number so its not dividing by 0
			val_split = numerator/(denominator+smallest_num)
			
			""" Get the entropy for both target feature 
				values with respect to this feature value
			"""
			# Add the smallest number so its not log2(0)
			val_entropy = val_entropy + -val_split*np.log2(val_split+smallest_num)

		# Get the entropy for all values in this feature
		val_ratio = denominator/len(heart)
		feature_entropy = feature_entropy + val_ratio*val_entropy
	
	return feature_entropy

def calc_info_gains(heart, info_gains):
	# Calculate the info_gain for non-target features only
	features = heart.drop(['target'], axis=1)

	# Get entropy of target feature
	target_entropy = get_target_entropy(heart)

	for f in features:
		feature_entropy = get_feature_entropy(heart, f)
		information_gain = target_entropy - feature_entropy
		info_gains[f] = information_gain

	return info_gains

def find_feature(heart, info_gains):
	info_gains = calc_info_gains(heart, info_gains)

	vals = list(info_gains.values())
	feat = list(info_gains.keys())

	return feat[vals.index(max(vals))]

def create_tree(heart, dec_tree = 0):
	# Find the feature to split on i.e. the node feature
	info_gains = {}
	node_feature = find_feature(heart, info_gains)
	node_feat_vals = heart[node_feature]

	# Initialise decision tree
	if dec_tree == 0:
		dec_tree = {}
		dec_tree[node_feature] = {}

	# Get all values for the node
	all_node_vals = np.unique(node_feat_vals)
	print(node_feature)
	# Build the tree with recursion
	for val in all_node_vals:
		sub_tree = heart[node_feat_vals == val].reset_index(drop=True)

		values, size = np.unique(sub_tree['target'], return_counts=True)
		print(val)
		print(values)
		print(size)
		# More of the tree needs to be built
		if len(size) > 1:
			print(dec_tree[node_feature])
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
	heart = load_dataframe()
	data = np.array([21,1,1,131,87,205,5,4,0,0,75,0])
	instance = pd.Series(data, index=['age','sex','cp','trestbps','trestbpd',
									'chol','cigs','years','fbs','famhist','thalrest',
									'exang'])

	print(instance)

	heart = heart.append(instance, ignore_index=True)

	# Bin features
	heart, columns_to_bin = bin_values(heart)
	columns_to_bin = ['age', 'trestbps', 'trestbpd', 'chol', 'cigs', 'years', 'thalrest']

	print(heart.tail())

	instance = heart.drop(['target'], axis=1).iloc[-1]
	heart = heart.drop(heart.index[-1])

	print(heart.tail())
	print(instance)
	# Build tree
	decision_tree = create_tree(heart)
	joblib.dump(decision_tree, 'decision_tree.pkl')
	# print(heart.iloc[4])

	# new_data = heart.drop(['target'], axis=1).iloc[4]

	# Make predictions
	pred = make_prediction(instance, decision_tree)

	print(pred)

main()