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

	heart_prediction_columns = [
		'age',
		'sex',
		'cp',
		'trestbps',
		'chol',
		'fbs',
		'target'
    ]

	heart_pred = heart[heart_prediction_columns]
	
	plt.matshow(heart_pred.corr())
	plt.xticks(np.arange(heart_pred.shape[1]), heart_pred.columns)
	plt.yticks(np.arange(heart_pred.shape[1]), heart_pred.columns)
	plt.colorbar()
	plt.show()

train_model()