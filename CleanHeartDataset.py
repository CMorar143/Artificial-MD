import os
import csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

path_heart = "../../FYP_Data/heart-disease-uci/"

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

# Take in a list of lists and convert it to one single list
def flatten(lst):
    for element in lst:
        if type(element) in (tuple, list):
            for i in flatten(element):
                yield i
        else:
            yield element

# Write the data to a new txt file
def write_new_file(final_list, dataset):
	# Create a new file
	new_cleveland = open(path_heart + 'new_' + dataset + '.txt', 'w')
	
	for entry in final_list:
		for value in entry:
			if 'name' in value:
				new_cleveland.write(value + '\n')
			else:
				new_cleveland.write(value + ', ')
	new_cleveland.close()


# Write the data to a new csv file
def write_csv_file(df, dataset):
	# df['dm'] = df['dm'].replace(to_replace=-9, value=0)
	df = df.replace(to_replace=-9, value=np.NaN)
	df.to_csv(path_heart + 'new_' + dataset + '.csv', encoding='utf-8', index=False)


# Clean up the input dataset so it can be used
def clean_text_file():
	# hungarian = open(path_heart + 'hungarian.txt', mode='r')
	# switzerland = open(path_heart + 'switzerland.txt', mode='r')
	# all_lines = hungarian.readlines()
	# dataset = 'hungarian'
	# all_lines = switzerland.readlines()
	# dataset = 'switzerland'

	# Read all the lines in the text file
	cleveland = open(path_heart + 'cleveland.txt', mode='r')
	all_lines = cleveland.readlines()
	dataset = 'cleveland'

	# For holding each instance of data
	new_list = []

	# This will contain a list of lists, each sublist
	# being one instance of data
	all_values = []

	# Loop through each value adding each value to a list
	# until the name column is reached. 
	# Once the name is reached then this list is added to the main list
	for index in range(0, len(all_lines)):
		new_list.append(all_lines[index].split(' '))

		if 'name' in all_lines[index]:
			all_values.append(new_list)
			new_list = []

	# This converts the list of lists to one single list
	all_values = list(flatten(all_values))

	final_list = []

	# Remove line breaks and create final list
	for i in range(0, len(all_values)):
		if '\n' in all_values[i]:
			all_values[i] = all_values[i][:-1]
		
		new_list.append(all_values[i])
		if 'name' in all_values[i]:
			final_list.append(new_list)
			new_list = []

	write_new_file(final_list, dataset)
	cleveland.close()

	return final_list, dataset


# Extract the parameters that will be used
def create_dataset():
	# Features
	heart_columns = {
		3: 'age',
		4: 'sex',
		9: 'cp',
		10: 'trestbps',
		37: 'trestbpd',
		12: 'chol',
		14: 'cigs',
		15: 'years',
		16: 'fbs',
		17: 'dm',
		18: 'famhist',
		33: 'thalrest',
		38: 'exang',
		58: 'target'
	}

	full_list, dataset = clean_text_file()
	all_input_params = []
	extracted_params = []

	# Loop through each instance in the full dataset and
	# extract the columns specified in the heart_columns dict
	for entry in full_list:
		for key in heart_columns:
			all_input_params.append(full_list[full_list.index(entry)][key-1])
		extracted_params.append(all_input_params)
		all_input_params = []
	
	# The target feature can have values ranging from 0-4
	# Use 'binning' so that the target value will either be 1 or 0
	for param in range(0, len(extracted_params)):
		extracted_params[param] = [float(i) for i in extracted_params[param]]
		if extracted_params[param][len(heart_columns)-1] > 0:
			extracted_params[param][len(heart_columns)-1] = 1.0

	# Create the dataframe and the csv file
	heart = pd.DataFrame(extracted_params, columns = list(heart_columns.values()))
	write_csv_file(heart, dataset)

	return extracted_params


create_dataset()

# Not being used
def process_csv():
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
