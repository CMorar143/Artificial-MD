import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# For data preparation
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import KNNImputer

# For building models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

# For evaluation
from yellowbrick.classifier import ClassificationReport
from sklearn.metrics import classification_report, confusion_matrix

def load_dataframe():
	# Load heart disease dataset into pandas dataframe
	pathHeart = "../../FYP_Data/heart-disease-uci/"
	heart = pd.read_csv(pathHeart + 'new_cleveland.csv')
	heart = heart.drop(['dm'], axis=1)
	print(heart.head())
	return heart


def plot_diagrams(heart):
	# Show correlation between features
	# plt.matshow(heart.corr())
	# plt.xticks(np.arange(heart.shape[1]), heart.columns)
	# plt.yticks(np.arange(heart.shape[1]), heart.columns)
	# plt.colorbar()
	# plt.show()
	# plt.close()

	# Show a histogram of all the columns
	heart.hist()
	plt.show()
	plt.close()	

	# Show the amount of entries who have and don't have heart disease
	plt.bar(['Does not have heart disease', 'Does have heart disease'], heart['target'].value_counts().sort_index(), color = ['blue', 'red'])
	plt.ylabel('Count')
	plt.show()


def scale_values(heart):
	heart = pd.get_dummies(heart, columns = ['cp'])
	columns_to_scale = ['age', 'trestbps', 'chol', 'cigs', 'years', 'thalrest', 'trestbpd']
	# min_max_scaler = preprocessing.MinMaxScaler()
	# heart[columns_to_scale] = min_max_scaler.fit_transform(heart[columns_to_scale])
	# return heart, min_max_scaler, columns_to_scale

	standardScaler = StandardScaler()
	heart[columns_to_scale] = standardScaler.fit_transform(heart[columns_to_scale])
	return heart, standardScaler, columns_to_scale


def impute_dataset(df, imputer):
	X = df.drop(['target'], axis = 1)
	Y = df['target']
	columns = X.columns
	X = imputer.fit_transform(X)
	df = pd.DataFrame(X, columns=columns)
	df2 = pd.DataFrame(Y, columns=['target'])
	df = df.reset_index(drop=True)
	df2 = df2.reset_index(drop=True)
	
	df = df.join(df2)

	return df


def scale_values_NN(X_train, X_test):
	standardScaler = StandardScaler()
	standardScaler.fit(X_train)

	X_train = standardScaler.transform(X_train)
	X_test = standardScaler.transform(X_test)

	return X_train, X_test


def split_dataset(X, D):
	return train_test_split(X, D, test_size = 0.28, random_state = 0)


def KNN(X_train, H_train, X_test, H_test):
	knn_scores = []
	max_score = 0
	knn_val = 0
	for k in range(1,21):
		knn_classifier = KNeighborsClassifier(n_neighbors = k)
		knn_classifier.fit(X_train, H_train)
		knn_scores.append(knn_classifier.score(X_test, H_test))
		if knn_classifier.score(X_test, H_test) > max_score:
			max_score = knn_classifier.score(X_test, H_test)
			knn_val = k

	plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
	for i in range(1,21):
		plt.text(i, knn_scores[i-1], (i, round(knn_scores[i-1], 2)))
	plt.xticks([i for i in range(1, 21)])
	plt.xlabel('Number of Neighbors (K)')
	plt.ylabel('Scores')
	plt.title('K Neighbors Classifier scores for different K values')
	plt.show()

	return max_score, knn_val


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

	return max(dt_scores)


def naive_bayes(X_train, H_train, X_test, H_test):
	model = GaussianNB()
	model.fit(X_train, H_train)
	test_pred = model.predict(X_test)
	
	visualizer = ClassificationReport(model, classes=['Negative','Positive'])
	visualizer.fit(X_train, H_train)
	visualizer.score(X_test, H_test)
	visualizer.poof()

	return metrics.accuracy_score(H_test, test_pred)


def linear_support_vector(X_train, H_train, X_test, H_test):
	svm_model = LinearSVC(random_state=0, max_iter=3500)
	svm_model.fit(X_train, H_train)
	test_pred = svm_model.predict(X_test)

	visualizer = ClassificationReport(svm_model, classes=['Negative','Positive'])
	visualizer.fit(X_train, H_train)
	visualizer.score(X_test, H_test)
	visualizer.poof()

	return metrics.accuracy_score(H_test, test_pred)


def display_accuracies(knn_acc, knn_val, dt_acc, nb_acc, lsv_acc):
	print(f'Accuracy of KNN value {knn_val}: {knn_acc}\n')
	print(f'Accuracy of Decision tree: {dt_acc}\n')
	print(f'Accuracy of Naive Bayes: {nb_acc}\n')
	print(f'Accuracy of LVM: {lsv_acc}\n')

	accuracies = [
		knn_acc, dt_acc, 
		nb_acc, lsv_acc
	]

	labels = [
		'K Nearest Neighbour', 'Decision Tree',
		'Naive Bayes', 'Linear Support Vector'
	]

	index = np.arange(len(labels))
	plt.bar(index, accuracies)
	# plt.xlabel('ML Model', fontsize=9)
	# plt.ylabel('Accuracy', fontsize=9)
	plt.xticks(index, labels, fontsize=9)
	plt.title('Accuracy for the different types of ML models')
	plt.show()


def build_NN():
	heart = load_dataframe()
	
	# Impute the remaining missing values
	imputer = KNNImputer(n_neighbors=3)
	heart = impute_dataset(heart, imputer)

	# Split dataset
	y = heart['target']
	X = heart.drop(['target'], axis = 1)
	X_train, X_test, y_train, y_test = split_dataset(X, y)
	X_train, X_test = scale_values_NN(X_train, X_test)

	# Build NN
	mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
	mlp.fit(X_train, y_train.values.ravel())
	predictions = mlp.predict(X_test)

	# Evaluate NN
	print(confusion_matrix(y_test,predictions))
	print(classification_report(y_test,predictions))
	print(mlp.n_iter_)
	print(mlp.classes_)
	print(mlp.loss)
	print(mlp.n_layers_)
	print(mlp.n_outputs_)
	print(mlp.out_activation_)

	


def train_heart_models():
	# Load dataframe
	heart = load_dataframe()

	# Impute the remaining missing values
	imputer = KNNImputer(n_neighbors=3)
	heart = impute_dataset(heart, imputer)

	# Use dummy columns for the categorical features
	# heart, scaler, columns_to_scale = scale_values(heart)

	# Split dataset
	H = heart['target']
	X = heart.drop(['target'], axis = 1)
	
	# With oversampling
	sm = SMOTE(random_state=52)
	x_sm, h_sm = sm.fit_sample(X, H)
	X_train, X_test, H_train, H_test = split_dataset(x_sm, h_sm)
	
	# Without Oversampling
	# X_train, X_test, H_train, H_test = split_dataset(X, H)
	X_train = X_train.round()
	# plot_diagrams(X_train)

	# KNN
	knn_acc, knn_val = KNN(X_train, H_train, X_test, H_test)

	# Decision Tree
	dt_acc = decision_tree(X_train, H_train, X_test, H_test, X)

	# Naive Bayes
	nb_acc = naive_bayes(X_train, H_train, X_test, H_test)

	# Linear Support Vector
	lsv_acc = linear_support_vector(X_train, H_train, X_test, H_test)

	display_accuracies(knn_acc, knn_val, dt_acc, nb_acc, lsv_acc)




	# Test the KNN classifier
	# knn_classifier_test = KNeighborsClassifier(n_neighbors = 8)
	# demo_values = [63, 1, 145, 85, 233, 50, 20, 1, 0, 1, 60, 0, 1, 0, 0, 0]
	# knn_classifier_test.fit(X_train, H_train)

	# df = pd.DataFrame(columns = X_test.columns) 
	# df.loc[0] = demo_values
	# print(df)

	# df[columns_to_scale] = scaler.transform(df[columns_to_scale])
	# print(df)

	# p = knn_classifier_test.predict(df)
	# print(p)


train_heart_models()
# build_NN()







