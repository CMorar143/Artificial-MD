# Start by importing libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def train_model():
	# Load heart disease dataset into pandas dataframe
	pathHeart = "../Data/heart-disease-uci/"
	
	heart = pd.read_csv(pathHeart + 'new_cleveland.csv')
	
	# hungarian = pd.read_csv(pathHeart + 'new_hungarian.csv')

	# switzerland = pd.read_csv(pathHeart + 'new_switzerland.csv')
	
	# to_combine = [
	# 	hungarian,
	# 	switzerland
	# ]

	# heart_columns = [
	# 	'age',
	# 	'sex',
	# 	'cp',
	# 	'trestbps',
	# 	'chol',
	# 	'cigs',
	# 	'years',
	# 	'fbs',
	# 	'dm',
	# 	'famhist',
	# 	'thalrest',
	# 	'trestbpd',
	# 	'exang',
	# 	'target'
	# ]

	# for df in to_combine:
		# heart =	pd.merge(heart, hungarian, on=['age', 'sex','cp', 'trestbps', 'chol', 'fbs', 'dm', 'famhist', 'thalrest', 'trestbpd', 'exang', 'target'], sort=True)
	
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
	plt.bar(['Does not have heart disease', 'Does have heart disease'], heart['target'].value_counts().sort_index(), color = ['blue', 'red'])
	plt.ylabel('Count')
	plt.show()

	# Use dummy columns for the categorical features
	heart = pd.get_dummies(heart, columns = ['sex', 'cp', 'fbs', 'dm', 'famhist', 'exang'])
	columns_to_scale = ['age', 'trestbps', 'chol', 'cigs', 'years', 'thalrest', 'trestbpd']
	standardScaler = StandardScaler()
	heart[columns_to_scale] = standardScaler.fit_transform(heart[columns_to_scale])

	y = heart['target']
	X = heart.drop(['target'], axis = 1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

	# KNN
	knn_scores = []
	for k in range(1,30):
		knn_classifier = KNeighborsClassifier(n_neighbors = k)
		knn_classifier.fit(X_train, y_train)
		knn_scores.append(knn_classifier.score(X_test, y_test))

	plt.plot([k for k in range(1, 30)], knn_scores, color = 'red')
	for i in range(1,30):
		plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
	plt.xticks([i for i in range(1, 30)])
	plt.xlabel('Number of Neighbors (K)')
	plt.ylabel('Scores')
	plt.title('K Neighbors Classifier scores for different K values')
	plt.show()

	# Decision Tree
	dt_scores = []
	for i in range(1, len(X.columns) + 1):
		dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
		dt_classifier.fit(X_train, y_train)
		dt_scores.append(dt_classifier.score(X_test, y_test))

	plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')
	for i in range(1, len(X.columns) + 1):
		plt.text(i, dt_scores[i-1], (i, dt_scores[i-1]))
	plt.xticks([i for i in range(1, len(X.columns) + 1)])
	plt.xlabel('Max features')
	plt.ylabel('Scores')
	plt.title('Decision Tree Classifier scores for different number of maximum features')
	plt.show()

	# Test the KNN classifier
	# knn_classifier_test = KNeighborsClassifier(n_neighbors = 8)
	# demo_values = [63, 145, 233, 150, 2.3, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
	# knn_classifier_test.fit(X_train, y_train)

	# df = pd.DataFrame(columns = X_test.columns) 
	# df.loc[0] = demo_values
	# print(df)

	# p = knn_classifier_test.predict(df)
	# print(p)




train_model()