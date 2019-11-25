# Start by importing libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def train_model():
	# Load heart disease dataset into pandas dataframe
	pathHeart = "../Data/heart-disease-uci/"
	heart = pd.read_csv(pathHeart + 'heart.csv')

	print(heart.columns)

	# # Remove columns that are not going to be used
	# heart_prediction_columns = [
	# 	'age',
	# 	'sex',
	# 	'cp',
	# 	'trestbps',
	# 	'chol',
	# 	'fbs',
	# 	'target'
	# ]

	# heart = heart[heart_prediction_columns]
	
	# Show correlation between features
	plt.matshow(heart.corr())
	plt.xticks(np.arange(heart.shape[1]), heart.columns)
	plt.yticks(np.arange(heart.shape[1]), heart.columns)
	plt.colorbar()
	plt.show()
	plt.close()

	# Show a histogram of all the columns
	heart.hist()
	plt.show()
	plt.close()
	
	# Show the amount of entries who have and don't have heart disease
	plt.bar(heart['target'].unique(), heart['target'].value_counts(), color = ['red', 'blue'])
	plt.xticks([0, 1])
	plt.xlabel('Target Classes')
	plt.ylabel('Count')
	plt.show()

	# 
	heart = pd.get_dummies(heart, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
	standardScaler = StandardScaler()
	columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
	heart[columns_to_scale] = standardScaler.fit_transform(heart[columns_to_scale])

	y = heart['target']
	X = heart.drop(['target'], axis = 1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)


train_model()