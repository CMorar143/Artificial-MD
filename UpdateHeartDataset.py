import os
import csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

path_heart = "../Data/heart-disease-uci/"

# 3 age: age in years
# 4 sex: sex (1 = male; 0 = female)
# 9 cp: chest pain type
# 	-- Value 1: typical angina
# 	-- Value 2: atypical angina
# 	-- Value 3: non-anginal pain
# 	-- Value 4: asymptomatic
# 10 trestbps: resting blood pressure (in mm Hg on admission to the hospital)
# 12 chol: serum cholestoral in mg/dl
# 14 cigs (cigarettes per day)
# 15 years (number of years as a smoker)
# 16 fbs: (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
# 17 dm (1 = history of diabetes; 0 = no such history)
# 18 famhist: family history of coronary artery disease (1 = yes; 0 = no)
# 33 thalrest: resting heart rate

# 37 trestbpd: resting blood pressure
# 38 exang: exercise induced angina (1 = yes; 0 = no)

# Input parameters
heart_columns = {
	3: 'age',
	4: 'sex',
	9: 'cp',
	10: 'trestbps',
	12: 'chol',
	14: 'cigs',
	15: 'years',
	16: 'fbs',
	17: 'dm',
	18: 'famhist',
	33: 'thalrest',
	37: 'trestbpd',
	38: 'exang',
	58: 'target'
}

# Take in a list of lists and convert it to one single list
def flatten(lst):
    for elem in lst:
        if type(elem) in (tuple, list):
            for i in flatten(elem):
                yield i
        else:
            yield elem

# Write the data to a new file
def write_new_file(final_list):
	# Create a new file
	new_cleveland = open(path_heart + 'new_cleveland.txt', 'w')
	
	for entry in final_list:
		for value in entry:
			if 'name' in value:
				new_cleveland.write(value + '\n')
			else:
				new_cleveland.write(value + ', ')
	new_cleveland.close()

# Clean up the input dataset so it can be used
def clean_text_file():
	cleveland = open(path_heart + 'cleveland.txt', mode='r')

	# Read all the lines in the text file
	all_lines = cleveland.readlines()

	new_list = []
	all_values = []
	final_list = []

	for index in range(0, len(all_lines)):
		new_list.append(all_lines[index].split(' '))

		if 'name' in all_lines[index]:
			all_values.append(new_list)
			new_list = []

	all_values = list(flatten(all_values))

	# Remove line breaks and create final list
	for i in range(0, len(all_values)):
		if '\n' in all_values[i]:
			all_values[i] = all_values[i][:-1]
		
		new_list.append(all_values[i])
		if 'name' in all_values[i]:
			final_list.append(new_list)
			new_list = []

	write_new_file(final_list)
	cleveland.close()

	return final_list


# Extract the parameters that will be used
def create_dataset():
	full_list = clean_text_file()
	all_input_params = []
	extracted_params = []

	# for i in range(0, len(full_list)):
	# 	if int(full_list[i][57]) > 1:
	# 		full_list[i][57] = 1

	for entry in full_list:
		for key in heart_columns:
			all_input_params.append(full_list[full_list.index(entry)][key-1])
		extracted_params.append(all_input_params)
		all_input_params = []

	zeroes = 0
	ones = 0
	twos = 0
	threes = 0
	
	# print("\n")
	# extracted_params = [i for list2 in extracted_params for float(item) in list2]
	for param in range(0, len(extracted_params)):
		extracted_params[param] = [float(i) for i in extracted_params[param]]
		if extracted_params[param][len(heart_columns)-1] > 0:
			extracted_params[param][len(heart_columns)-1] = 1.0

			if extracted_params[param][len(heart_columns)-1] == 1:
				ones = ones + 1
			if extracted_params[param][len(heart_columns)-1] == 2:
				twos = twos + 1
			if extracted_params[param][len(heart_columns)-1] == 3:
				threes = threes + 1
		else:
			zeroes = zeroes + 1

	print(f"the number of zeroes is: {zeroes}")
	print(f"the number of ones is: {ones}")
	print(f"the number of twos is: {twos}")
	print(f"the number of threes is: {threes}")
	print("\n")
	return extracted_params


def train_model():
	extracted_params = create_dataset()
	
	heart = pd.DataFrame(extracted_params, columns = list(heart_columns.values()))

	# print(heart.head())
	zeroes = heart[heart.target == 0].shape[0]
	ones = heart[heart.target == 1].shape[0]
	twos = heart[heart.target == 2].shape[0]
	threes = heart[heart.target == 3].shape[0]

	print(f"the number of zeroes is: {zeroes}")
	print(f"the number of ones is: {ones}")
	print(f"the number of twos is: {twos}")
	print(f"the number of threes is: {threes}")

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
	plt.bar(['Does not have heart disease', 'Does have heart disease'], heart['target'].value_counts(), color = ['blue', 'red'])
	plt.ylabel('Count')
	plt.show()

	# Use dummy columns for the categorical features
	# Also use the 
	# heart = pd.get_dummies(heart, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
	heart = pd.get_dummies(heart, columns = ['sex', 'cp', 'fbs', 'dm', 'famhist', 'exang'])
	standardScaler = StandardScaler()
	# columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
	columns_to_scale = ['age', 'trestbps', 'chol', 'cigs', 'years', 'thalrest', 'trestbpd']
	heart[columns_to_scale] = standardScaler.fit_transform(heart[columns_to_scale])


	y = heart['target']
	X = heart.drop(['target'], axis = 1)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

	# KNN
	knn_scores = []
	for k in range(1,21):
		knn_classifier = KNeighborsClassifier(n_neighbors = k)
		knn_classifier.fit(X_train, y_train)
		knn_scores.append(knn_classifier.score(X_test, y_test))

	plt.plot([k for k in range(1, 21)], knn_scores, color = 'red')
	for i in range(1,21):
		plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
	plt.xticks([i for i in range(1, 21)])
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




def is_num(string):
	try:
		float(string)
		return True
	except ValueError:
		return False

train_model()
# process_csv()

# Not being used
def process_csv():
	# pathHeart = "../Data/heart-disease-uci/"

	# Read in dataset
	heart = pd.read_csv(path_heart + 'heart.csv')

	# Read in all the original datasets
	cleveland = pd.read_csv(path_heart + 'cleveland.txt', header=None).replace(' ', ',', regex=True)
	hungarian = pd.read_csv(path_heart + 'hungarian.txt', header=None).replace(' ', ',', regex=True)
	switzerland = pd.read_csv(path_heart + 'switzerland.txt', header=None).replace(' ', ',', regex=True)

	cleveland = cleveland.replace(r'\s', ' ', regex=True)
	print(cleveland)

	cleveland.close()
	hungarian.close()
	switzerland.close()
