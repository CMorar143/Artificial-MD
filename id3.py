import numpy as np
import pandas as pd

def load_dataframe():
	# Load heart disease dataset into pandas dataframe
	pathHeart = "../../FYP_Data/heart-disease-uci/"
	heart = pd.read_csv(pathHeart + 'new_cleveland.csv')
	heart = heart.drop(['dm'], axis=1)
	print(heart.head())
	return heart

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

def main(heart, dec_tree = 0):
	# Find the feature to split on i.e. the node feature
	info_gains = {}
	node_feature = find_feature(heart, info_gains)

	# Initialise decision tree
	if dec_tree == 0:
		dec_tree = {}
		dec_tree[node_feature] = {}

	# Get all values for the root node
	all_node_vals = np.unique(heart[node_feature])

	# Build the tree with recursion
	for val in all_node_vals:
		sub_tree = heart[heart[node_feature] == val].reset_index(drop=True)

		values, size = np.unique(sub_tree['target'],return_counts=True)
		
		# More of the tree needs to be built
		if len(size) > 1:
			dec_tree[node_feature][val] = main(sub_tree) 
		
		# This is the leaf node
		else:
			dec_tree[node_feature][val] = values[0]
			
	return dec_tree


# Load dataset
heart = load_dataframe()

# Build tree
decision_tree = main(heart)
