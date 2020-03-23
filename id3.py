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

	# Values are they have heart disease or they don't (1 or 0 respectively)
	values = heart['target'].unique()
	feature_vals = heart[feature].unique()

	for value in feature_vals:
		for val in values:
			val_split = heart['target'].value_counts()[val]/len(heart['target'])
			entropy = entropy + -val_split*np.log2(val_split)

	return entropy

# Load dataset
heart = load_dataframe()

# Get entropy of target feature
target_entropy = get_target_entropy(heart)

feature = 'sex'
print(heart[feature].value_counts())
num = len(heart[feature][heart[feature]==0.0][heart['target'] ==1.0])
print(num)
print(heart[feature][heart[feature]==0.0])
# Next we find the entropy of every other feature
