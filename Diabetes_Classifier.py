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

data_path = "../../FYP_Data/Health_Survey/"

def load_dataframe():
	# Load diabetes dataset into pandas dataframe
	# data_path = "../../FYP_Data/Health_Survey/"
	diabetes = pd.read_csv(data_path + 'Diabetes.csv')

	return diabetes



def plot_diagrams(diabetes):
	# Show correlation between features
	plt.matshow(diabetes.corr())
	plt.xticks(np.arange(diabetes.shape[1]), diabetes.columns)
	plt.yticks(np.arange(diabetes.shape[1]), diabetes.columns)
	plt.colorbar()
	plt.show()
	plt.close()

	# Show a histogram of all the columns
	diabetes.hist()
	plt.show()
	plt.close()	

	# Show the amount of entries who have and don't have diabetes
	plt.bar(['Does not have diabetes', 'Does have diabetes'], diabetes['Diabetes'].value_counts().sort_index(), color = ['blue', 'red'])
	plt.ylabel('Count')
	plt.show()


def scale_values(diabetes):
	columns_to_scale = ['BMI', 'Sys_BP', 'Dias_BP', 'HDL_Chol', 'LDL_Chol', 'Total_Chol', 'Fast_Glucose', 'Triglyceride', 'Uric_Acid']
	min_max_scaler = preprocessing.MinMaxScaler()
	diabetes[columns_to_scale] = min_max_scaler.fit_transform(diabetes[columns_to_scale])

	diabetes.to_csv(data_path + "Diabetes_scaled.csv", index=False)

	return diabetes


def scale_values_NN(X_train, X_test):
	standardScaler = StandardScaler()
	standardScaler.fit(X_train)

	X_train = standardScaler.transform(X_train)
	X_test = standardScaler.transform(X_test)

	return X_train, X_test


def split_dataset(X, D):
	return train_test_split(X, D, test_size = 0.25, random_state = 0)


def KNN(X_train, D_train, X_test, D_test):
	knn_scores = []
	k_max = 21
	for k in range(1, k_max):
		knn_classifier = KNeighborsClassifier(n_neighbors = k)
		knn_classifier.fit(X_train, D_train)
		knn_scores.append(knn_classifier.score(X_test, D_test))

	plt.plot([k for k in range(1, k_max)], knn_scores, color = 'red')
	for i in range(1, k_max):
		plt.text(i, knn_scores[i-1], (i, round(knn_scores[i-1], 2)))
	plt.xticks([i for i in range(1, k_max)])
	plt.xlabel('Number of Neighbors (K)')
	plt.ylabel('Scores')
	plt.title('K Neighbors Classifier scores for different K values')
	plt.show()

	knn_classifier = KNeighborsClassifier(n_neighbors = 4)
	knn_classifier.fit(X_train, D_train)

	visualizer = ClassificationReport(knn_classifier, classes=['Negative','Positive'])
	visualizer.fit(X_train, D_train)
	visualizer.score(X_test, D_test)
	visualizer.poof()


def decision_tree(X_train, D_train, X_test, D_test, X):
	dt_scores = []
	for i in range(1, len(X.columns) + 1):
		dt_classifier = DecisionTreeClassifier(max_features = i, random_state = 0)
		dt_classifier.fit(X_train, D_train)
		dt_scores.append(dt_classifier.score(X_test, D_test))

	plt.plot([i for i in range(1, len(X.columns) + 1)], dt_scores, color = 'green')
	for i in range(1, len(X.columns) + 1):
		plt.text(i, dt_scores[i-1], (i, round(dt_scores[i-1], 2)))
	plt.xticks([i for i in range(1, len(X.columns) + 1)])
	plt.xlabel('Max features')
	plt.ylabel('Scores')
	plt.title('Decision Tree Classifier scores for different number of maximum features')
	plt.show()

	dt_classifier = DecisionTreeClassifier(max_features = 6, random_state = 0)
	dt_classifier.fit(X_train, D_train)

	visualizer = ClassificationReport(dt_classifier, classes=['Negative','Positive'])
	visualizer.fit(X_train, D_train)
	visualizer.score(X_test, D_test)
	visualizer.poof()


def naive_bayes(X_train, D_train, X_test, D_test):
	model = GaussianNB()
	model.fit(X_train, D_train)
	test_pred = model.predict(X_test)
	print(f'Accuracy of NB: {metrics.accuracy_score(D_test, test_pred)}\n')

	visualizer = ClassificationReport(model, classes=['Negative','Positive'])
	visualizer.fit(X_train, D_train)
	visualizer.score(X_test, D_test)
	visualizer.poof()



def linear_support_vector(X_train, D_train, X_test, D_test):
	svm_model = LinearSVC(random_state=0, max_iter=10000)
	svm_model.fit(X_train, D_train)
	test_pred = svm_model.predict(X_test)
	print(f'Accuracy of LVM: {metrics.accuracy_score(D_test, test_pred)}\n')

	visualizer = ClassificationReport(svm_model, classes=['Negative','Positive'])
	visualizer.fit(X_train, D_train)
	visualizer.score(X_test, D_test)
	visualizer.poof()


def build_NN():
	diabetes = load_dataframe()

	# Split dataset
	y = diabetes['Diabetes']
	X = diabetes.drop(['Diabetes'], axis = 1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
	X_train, X_test = scale_values_NN(X_train, X_test)

	# Build NN
	mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
	mlp.fit(X_train, y_train.values.ravel())
	predictions = mlp.predict(X_test)

	# Evaluate NN
	print(confusion_matrix(y_test,predictions))
	print(classification_report(y_test,predictions))
	

def train_diabetes_models():
	# Load dataframe
	diabetes = load_dataframe()
	plot_diagrams(diabetes)

	# Normalise values
	diabetes = scale_values(diabetes)

	# Split dataset
	D = diabetes['Diabetes']
	X = diabetes.drop(['Diabetes'], axis = 1)

	# With oversampling
	sm = SMOTE(random_state=52)
	x_sm, d_sm = sm.fit_sample(X, D)
	X_train, X_test, D_train, D_test = split_dataset(x_sm, d_sm)
	
	# Without Oversampling
	# X_train, X_test, D_train, D_test = split_dataset(X, D)

	# KNN
	KNN(X_train, D_train, X_test, D_test)

	# Decision Tree
	decision_tree(X_train, D_train, X_test, D_test, X)

	# Naive Bayes
	naive_bayes(X_train, D_train, X_test, D_test)

	# Linear Support Vector
	linear_support_vector(X_train, D_train, X_test, D_test)

	# Test the KNN classifier
	# knn_classifier_test = KNeighborsClassifier(n_neighbors = 8)
	# demo_values = [63, 145, 233, 150, 2.3, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]
	# knn_classifier_test.fit(X_train, D_train)

	# df = pd.DataFrame(columns = X_test.columns)
	# df.loc[0] = demo_values
	# print(df)

	# p = knn_classifier_test.predict(df)
	# print(p)

train_diabetes_models()
# build_NN()







# import pandas as pd
# from sklearn import preprocessing
# from imblearn.over_sampling import SMOTE
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier

# # Requires scikit-learn v0.22
# from sklearn.impute import KNNImputer

# def process_dataset(df):
# 	categorical_features = ['job', 'marital', 'education', 'default',  'housing', 'loan', 'contact', 'month', 'poutcome']
# 	columns_to_scale = ['age', 'balance', 'day', 'campaign', 'pdays', 'previous']

# 	# Replace instances of 'unknown' with NaN so it won't consider missing values as cardinality
# 	df.replace(to_replace = 'unknown', value = np.NaN, inplace = True)
# 	df['poutcome'].replace(to_replace = np.NaN, value = 'unknown', inplace = True)

# 	# drop the id columns
# 	df = df.drop(['id'], axis=1)

# 	# Turn categorical features into numerical
# 	encoded_dict = {}
# 	for feature in categorical_features:
# 		cats = pd.Categorical(df[feature]).categories
# 		d = {}
# 		for i, cat in enumerate(cats):
# 			d[cat] = i
# 		encoded_dict[feature] = d

# 	for k, v in encoded_dict.items():
# 		df[k].replace(encoded_dict[k], inplace=True, regex=True)

# 	# Scale continuous variables
# 	to_norm = df[columns_to_scale].values.astype(float)
# 	min_max_scaler = preprocessing.MinMaxScaler()
# 	df[columns_to_scale] = min_max_scaler.fit_transform(to_norm)

# 	return df


# def impute_dataset(df, imputer):
# 	X = df.drop(['target'], axis = 1)
# 	columns = X.columns
# 	X = imputer.fit_transform(X)
# 	df = pd.DataFrame(X, columns=columns)
# 	df['contact'] = df['contact'].round()

# 	return X, df


# def build_classifier(X_train, Y_train):
# 	k = 3
# 	knn_classifier = KNeighborsClassifier(n_neighbors = k)
# 	knn_classifier.fit(X_train, Y_train)

# 	return knn_classifier


# def main():
# 	Features = [
# 		'id',
# 		'age',
# 		'job',
# 		'marital',
# 		'education',
# 		'default',
# 		'balance',
# 		'housing',
# 		'loan',
# 		'contact',
# 		'day',
# 		'month',
# 		'campaign',
# 		'pdays',
# 		'previous',
# 		'poutcome',
# 		'target'
# 	]

# 	# Load in the dataset and the feature names
# 	df = pd.read_csv('trainingset.csv', header = None)
# 	df.columns = Features

# 	# Clean dataset
# 	df = process_dataset(df)
# 	print("\nDataset has been processed.\nImputing features...\n")
# 	# Impute features
# 	imputer = KNNImputer(n_neighbors=3)

# 	# Split training features and target
# 	Y = df['target']
# 	X, df = impute_dataset(df, imputer)
# 	print('Values have been imputed.\nBuilding Classifier...\n')

# 	# Perform oversampling
# 	sm = SMOTE(random_state=52)
# 	x_train_res, y_train_res = sm.fit_sample(X, Y)
# 	knn_classifier = build_classifier(x_train_res, y_train_res)
# 	print("Classifier Built.\nMaking prediction...")

# 	############### PREDICTING ##################

# 	# Read in queries data
# 	queries = pd.read_csv('queries.csv', header = None)
# 	queries.columns = Features

# 	# Save for later
# 	ids = queries['id']

# 	# Clean dataset
# 	queries = process_dataset(queries)

# 	# Split training features and target
# 	_, queries = impute_dataset(queries, imputer)

# 	# Make prediction
# 	predictions = knn_classifier.predict(queries)

# 	# Join predictions with ids
# 	predictions = list(zip(ids, predictions))

# 	# Write to csv file
# 	predictions_file = 'Predictions_KNN.csv'
# 	predictions = pd.DataFrame(predictions, columns=None)
# 	predictions.to_csv(predictions_file, index=False, header=None)

# 	# Remove blank line at the end of the file
# 	predictions = open(predictions_file, 'rb').read()
# 	open(predictions_file, 'wb').write(predictions[:-2])


# main()
