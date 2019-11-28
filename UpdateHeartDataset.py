import os
import csv
import pandas as pd

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
	10: 'trestbpd',
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

	# Iterate through the individual rows
	# for _, row in cleveland.iterrows():
	# 	print(row[0].replace('\n', ','))

	cleveland.close()
	hungarian.close()
	switzerland.close()

def flatten(lst):
    for elem in lst:
        if type(elem) in (tuple, list):
            for i in flatten(elem):
                yield i
        else:
            yield elem

def text_file():
	cleveland = open(path_heart + 'cleveland.txt', mode='r')

	# Read all the lines in the text file
	all_lines = cleveland.readlines()
	new_list = []
	final_list = []
	f_list = []

	for index in range(0, len(all_lines)):
		new_list.append(all_lines[index].split(' '))

		if 'name' in all_lines[index]:
			final_list.append(new_list)
			new_list = []

	final_list = list(flatten(final_list))

	# Remove all line breaks from the dataset
	for i in range(0, len(final_list)):
		if '\n' in final_list[i]:
			final_list[i] = final_list[i][:-1]
		new_list.append(final_list[i])

		if 'name' in final_list[i]:
			f_list.append(new_list)
			new_list = []



	# for i in range(0, len(final_list)):
	# 	final_list[i] = list(flatten(final_list[i]))

	print(f_list[0])

	# Create a new file
	new_cleveland = open(path_heart + 'new_cleveland.txt', 'w')

	for line in all_lines:
		for word in line:
			new_cleveland.write(word + ', ')
	
	new_cleveland.close()
	cleveland.close()


def is_num(string):
	try:
		float(string)
		return True
	except ValueError:
		return False

text_file()
# process_csv()
