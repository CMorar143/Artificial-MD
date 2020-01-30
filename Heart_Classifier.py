# Start by importing libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


def load_dataframe():
	# Load heart disease dataset into pandas dataframe
	pathHeart = "../../FYP_Data/heart-disease-uci/"
	heart = pd.read_csv(pathHeart + 'new_cleveland.csv')

	return heart


def plot_diagrams(heart):
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


def scale_values(heart):
	heart = pd.get_dummies(heart, columns = ['sex', 'cp', 'fbs', 'dm', 'famhist', 'exang'])
	columns_to_scale = ['age', 'trestbps', 'chol', 'cigs', 'years', 'thalrest', 'trestbpd']
	standardScaler = StandardScaler()
	heart[columns_to_scale] = standardScaler.fit_transform(heart[columns_to_scale])

	return heart


def KNN(X_train, H_train, X_test, H_test):
	knn_scores = []
	for k in range(1,30):
		knn_classifier = KNeighborsClassifier(n_neighbors = k)
		knn_classifier.fit(X_train, H_train)
		knn_scores.append(knn_classifier.score(X_test, H_test))

	plt.plot([k for k in range(1, 30)], knn_scores, color = 'red')
	for i in range(1,30):
		plt.text(i, knn_scores[i-1], (i, round(knn_scores[i-1], 2)))
	plt.xticks([i for i in range(1, 30)])
	plt.xlabel('Number of Neighbors (K)')
	plt.ylabel('Scores')
	plt.title('K Neighbors Classifier scores for different K values')
	plt.show()


def decision_tree(X_train, H_train, X_test, H_test, X):
	dt_scores = []
	for i in range(1, len(X.columns) + 1):
		dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
		dt_classifier.fit(X_train, H_train)
		dt_scores.append(dt_classifier.score(X_test, H_test))

	plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')
	for i in range(1, len(X.columns) + 1):
		plt.text(i, dt_scores[i-1], (i, round(dt_scores[i-1], 2)))
	plt.xticks([i for i in range(1, len(X.columns) + 1)])
	plt.xlabel('Max features')
	plt.ylabel('Scores')
	plt.title('Decision Tree Classifier scores for different number of maximum features')
	plt.show()


def naive_bayes(X_train, H_train, X_test, H_test):
	model = GaussianNB()
	model.fit(X_train, H_train)
	test_pred = model.predict(X_test)
	print(f'Accuracy of NB: {metrics.accuracy_score(H_test, test_pred)}')


def train_heart_models():
	# Load dataframe
	heart = load_dataframe()
	plot_diagrams(heart)

	# Use dummy columns for the categorical features
	heart = scale_values(heart)

	# Split dataset
	H = heart['target']
	X = heart.drop(['target'], axis = 1)
	X_train, X_test, H_train, H_test = train_test_split(X, H, test_size = 0.33, random_state = 0)

	# KNN
	KNN(X_train, H_train, X_test, H_test)

	# Decision Tree
	decision_tree(X_train, H_train, X_test, H_test, X)

	# Naive Bayes
	naive_bayes(X_train, H_train, X_test, H_test)

	# Test the KNN classifier
	# knn_classifier_test = KNeighborsClassifier(n_neighbors = 8)
	# demo_values = [63, 145, 233, 150, 2.3, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
	# knn_classifier_test.fit(X_train, H_train)

	# df = pd.DataFrame(columns = X_test.columns) 
	# df.loc[0] = demo_values
	# print(df)

	# p = knn_classifier_test.predict(df)
	# print(p)


train_heart_models()