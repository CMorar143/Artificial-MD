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
			heart[col] = pd.cut(heart[col], 10, include_lowest=True)
		else:
			heart[col] = pd.cut(heart[col], 7)
	
	# heart = pd.get_dummies(heart, columns = columns_to_bin)

	return heart

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
	# data = np.array([29,1,1,131,87,205,5,4,0,0,75,0])
	# instance = pd.Series(data, index=['age','sex','cp','trestbps','trestbpd',
	# 								'chol','cigs','years','fbs','famhist','thalrest',
	# 								'exang'])

	# print(instance)

	# heart = heart.append(instance, ignore_index=True)

	# Bin features
	heart = bin_values(heart)

	print(heart.tail())

	instance = heart.drop(['target'], axis=1).iloc[-1]
	heart = heart.drop(heart.index[-1])

	







	# for col in columns_to_bin:
	# 	print(heart[col].value_counts())
	# 	# print(instance[col])
	# 	print("\n\n\n")
	# 	values = np.unique(heart[col])
	# 	print(values)
	# 	print(instance[col])
	# 	print("\n\n\n")
	# 	if instance[col] in values:
	# 		print(col)

	# 	print("\n\n\n")
	# 	print("\n\n\n")

	print(heart.tail())
	print(instance)
	# Build tree
	
	decision_tree = joblib.load('decision_tree.pkl')
	print(decision_tree)
	
		# print(heart.iloc[4])

	# new_data = heart.drop(['target'], axis=1).iloc[4]

	# Make predictions
	pred = make_prediction(instance, decision_tree)

	print(pred)

main()