import os
import csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.impute import KNNImputer
from imblearn.over_sampling import SMOTE

def impute_dataset(df, imputer):
	X = df.drop(['Diabetes'], axis = 1)
	Y = df['Diabetes']
	columns = X.columns
	X = imputer.fit_transform(X)
	df = pd.DataFrame(X, columns=columns)
	df2 = pd.DataFrame(Y, columns=['Diabetes'])
	df = df.reset_index(drop=True)
	df2 = df2.reset_index(drop=True)
	
	df = df.join(df2)

	return df

def process_dir():
	path = "../../FYP_Data/Health_Survey/"
	flist = os.listdir(path)
	featureNames = []

	# demographic = pd.read_csv(path + 'demographic.csv')
	diet = pd.read_csv(path + 'diet.csv')
	examination = pd.read_csv(path + 'examination.csv')
	labs = pd.read_csv(path + 'labs.csv')
	questionnaire = pd.read_csv(path + 'questionnaire.csv')
	glucose = pd.read_csv(path + 'GLU_H.csv')

	df_list = [
		diet,
		examination,
		labs,
		glucose,
		questionnaire
	]

	Cont_Columns = [
		'Count',
		'% Miss.',
		'Card.',
		'Min.',
		'1st Qrt.',
		'Mean',
		'Median',
		'3rd Qrt.',
		'Max',
		'Std. Dev.'
	]

	Cat_Columns = [
		'Count',
		'% Miss.',
		'Card.',
		'Mode',
		'Mode Freq.',
		'Mode %',
		'2nd Mode',
		'2nd Mode Freq.',
		'2nd Mode %'
	]

	Features = [
		'Short_Breath',
		'Chest_Pains',
		'High_Chol_Hist',
		'High_BP_Hist',
		'BMI',
		'Reg_Pulse',
		'Pulse_Type',
		'Sys_BP',
		'Dias_BP',
		'Protein',
		'HDL_Chol',
		'LDL_Chol',
		'Total_Chol',
		'Fast_Glucose',
		'Triglyceride',
		'Uric_Acid',
		'Diabetes'
	]

	Bool_Features = [
		'Short_Breath',
		'Chest_Pains',
		'High_Chol_Hist',
		'High_BP_Hist',
		'Reg_Pulse',
		'Diabetes'
	]

	input_params = [
		'CDQ010',
		'CDQ001',
		'BPQ080',
		'BPQ020',
		'BMXBMI',
		'BPXPULS',
		'BPXPTY',
		'BPXSY1',
		'BPXDI1',
		'LBXSTP',
		'LBDHDD',
		'LBDLDL',
		'LBXTC',
		'LBXGLU',
		'LBXTR',
		'LBXSUA',
		'DIQ010'
	]

	df_combined = pd.DataFrame.copy(diet)

	for f in df_list[1:]:
		df_combined = pd.merge(df_combined, f, on='SEQN', sort=False)
	
	df_combined.set_index('SEQN', inplace=True)
	df_input_params = df_combined[input_params]

	df_input_params.columns = Features

	for col in Bool_Features:
		df_input_params[col].replace(to_replace=1.0, value=0, inplace=True)
		df_input_params[col].replace(to_replace=2.0, value=1, inplace=True)
		df_input_params[col].replace(to_replace=9.0, value=np.NaN, inplace=True)

	df_input_params['Diabetes'].replace(to_replace=3.0, value=1.0, inplace=True)

	dropna_features = []
	# for col in range(2, len(Features)):
	for col in range(2):
		dropna_features.append(Features[col])

	# print(dropna_features)
	# df_input_params.dropna(subset=dropna_features, inplace=True)
	df_input_params.dropna(thresh=15, inplace=True)
	
	# Impute the remaining missing values
	imputer = KNNImputer(n_neighbors=3)
	# df_input_params = impute_dataset(df_input_params, imputer)

	# round the imputed values for dichotomous features
	# print(df_input_params['High_BP_Hist'].value_counts())
	# df_input_params['High_BP_Hist'] = df_input_params['High_BP_Hist'].round()
	# print(df_input_params['High_BP_Hist'].value_counts())

	# print(df_input_params['High_Chol_Hist'].value_counts())
	# df_input_params['High_Chol_Hist'] = df_input_params['High_Chol_Hist'].round()
	# print(df_input_params['High_Chol_Hist'].value_counts())

	# print(df_input_params['Reg_Pulse'].value_counts())
	# df_input_params['Reg_Pulse'] = df_input_params['Reg_Pulse'].round()
	# print(df_input_params['Reg_Pulse'].value_counts())
	
	# df_input_params.to_csv(path + "Diabetes.csv", index=False)




	# For Continous features
	min_value = 0
	first_qrt = 0
	mean = 0
	median = 0
	third_qrt = 0
	max_value = 0
	stand_dev = 0

	# For Both
	count = 0
	perc_missing = 0
	count_missing = 0
	card = 0

	d = dict(df_input_params.apply(pd.Series.nunique))
	count = len(df_input_params)

	for i in Features:
		# Count
		array = df_input_params[i]

		d2 = dict(array.value_counts())
		# count_missing = d[' ?']

		# % Missing
		array2 = set(array)
		count_missing = array.isna().sum()

		if count_missing == 0:
			perc_missing = 0
		else:
			perc_missing = (count_missing / count) * 100

		# Cardinality
		card = d[i]

		# Minimum
		min_value = array.min()

		# First Quartile
		first_qrt = array.quantile(0.25)

		# Mean
		mean = array.mean()

		# Median
		median = array.median()

		# Third Quartile
		third_qrt = array.quantile(0.75)

		# Maximum
		max_value = array.max()

		# Standard Deviation
		stand_dev = array.std()

		# print(array.value_counts())
		# print("\n")
		print(i)
		print("count", count)
		# print("count_missing", count_missing)
		print("perc_missing", perc_missing)
		print("card", card)
		# print("min_value", min_value)
		# print("first_qrt", first_qrt)
		# print("mean", mean)
		# print("median", median)
		# print("third_qrt", third_qrt)
		# print("max_value", max_value)
		# print("stand_dev", stand_dev)
		# print("mode:",mode)
		# print("\n")
		# print("mode_freq:",mode_freq)
		# print("\n")
		# print("mode_perc:",mode_perc)
		# print("\n")
		# print("mode2:",mode2)
		# print("\n")
		# print("mode2_freq:",mode2_freq)
		# print("\n")
		# print("mode2_perc:",mode2_perc)
		print("\nNEXT\n")


def main():
	process_dir()

main()