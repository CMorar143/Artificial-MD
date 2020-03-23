import numpy as np
import pandas as pd

def load_dataframe():
	# Load heart disease dataset into pandas dataframe
	pathHeart = "../../FYP_Data/heart-disease-uci/"
	heart = pd.read_csv(pathHeart + 'new_cleveland.csv')
	heart = heart.drop(['dm'], axis=1)
	print(heart.head())
	return heart

def get_df_entropy(heart):
	# Get target class
	target = heart.keys()[-1]
	entropy = 0

	# Values are they have heart disease or they don't (1 or 0 respectively)
	values = heart[target].unique()

	for value in values:
		val_split = heart[target].value_counts()[value]/len(heart[target])
		print(val_split)
		
	return entropy


heart = load_dataframe()

print(heart)

entropy = get_df_entropy(heart)
print(entropy)