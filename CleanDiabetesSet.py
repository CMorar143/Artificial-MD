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

def Create_DataFrame(Features, Columns, Listofvals):
    dataF = pd.DataFrame(index = Features)

    # Loop through the list of continous/categorical columns
    # and append each one to the DataFrame
    counter = 0
    for i in Columns:
        dataF[i] = Listofvals[counter]
        counter = counter + 1

    return dataF

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

	Cont_Features = [
		'BMI',
		'Sys_BP',
		'Dias_BP',
		'Protein',
		'HDL_Chol',
		'LDL_Chol',
		'Total_Chol',
		'Fast_Glucose',
		'Triglyceride',
		'Uric_Acid',
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

	Cat_Features = [
		'Short_Breath',
		'Chest_Pains',
		'High_Chol_Hist',
		'High_BP_Hist',
		'Pulse_Type',
		'Diabetes'
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
		'Diabetes'
	]

	input_params = [
		'CDQ010',
		'CDQ001',
		'BPQ080',
		'BPQ020',
		'BMXBMI',
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
	# df_input_params.dropna(thresh=15, inplace=True)
	
	# Impute the remaining missing values
	# imputer = KNNImputer(n_neighbors=3)
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

	# Next we instantiate lists to contain each of the columns values
	# For Categorical features
	All_Cat_vals = []
	Lmode = []
	Lmode_perc = []
	Lmode_freq = []
	Lmode2 = []
	Lmode2_perc = []
	Lmode2_freq = []

	# For Continous features
	All_Cont_vals = []
	Lmin_value = []
	Lfirst_qrt = []
	Lmean = []
	Lmedian = []
	Lthird_qrt = []
	Lmax_value = []
	Lstand_dev = []

	# For Both
	Lperc_missing = []
	Lcard = []
	total_instances = len(df_input_params)

	# For checking cardinality
	d = dict(df_input_params.apply(pd.Series.nunique))


	################### Categorical Features ###################

	for i in Cat_Features:
		# Create a ndarray for this column
		feature_values = df_input_params[i]

		# Get the total number of missing values
		count_missing = feature_values.isna().sum()

		# Remove white space
		# feature_values = feature_values.str.lstrip()

		# To contain values and their frequencies in descending order
		d2 = dict(feature_values.value_counts())

		# Count
		count = feature_values.count() + count_missing

		# % Missing
		perc_missing = (count_missing / total_instances) * 100
		Lperc_missing.append(perc_missing)

		# Cardinality
		card = d[i]
		Lcard.append(card)

		# Mode
		mode = list(d2)[0]
		Lmode.append(mode)

		# Mode Freq.
		mode_freq = d2[mode]
		Lmode_freq.append(mode_freq)

		# Mode %
		mode_perc = (mode_freq / count) * 100
		Lmode_perc.append(mode_perc)

		# 2nd Mode
		mode2 = list(d2)[1]
		Lmode2.append(mode2)

		# 2nd Mode Freq.
		mode2_freq = d2[mode2]
		Lmode2_freq.append(mode2_freq)

		# 2nd Mode %
		mode2_perc = (mode2_freq / count) * 100
		Lmode2_perc.append(mode2_perc)

	# Add all lists to a list
	# This will be used for creating the dataframe
	All_Cat_vals.extend([
	        count, Lperc_missing, Lcard,
	        Lmode, Lmode_freq, Lmode_perc,
	        Lmode2, Lmode2_freq, Lmode2_perc
	    ])
	print("before create")
	Cat_df = Create_DataFrame(Cat_Features, Cat_Columns, All_Cat_vals)
	print("after create")

	################### Continous Features ###################

	# Clear these lists as they are needed for Continous features as well
	Lperc_missing.clear()
	Lcard.clear()

	for i in Cont_Features:
		# Create a ndarray for this column
		feature_values = df_input_params[i]

		# Get the total number of missing values
		count_missing = feature_values.isna().sum()

		# Count
		count = feature_values.count() + count_missing

		# % Missing
		perc_missing = (count_missing / total_instances) * 100
		Lperc_missing.append(perc_missing)

		# Cardinality
		card = d[i]
		Lcard.append(card)

		# Minimum
		min_value = feature_values.min()
		Lmin_value.append(min_value)

		# First Quartile
		first_qrt = feature_values.quantile(0.25)
		Lfirst_qrt.append(first_qrt)

		# Mean
		mean = feature_values.mean()
		Lmean.append(mean)

		# Median
		median = feature_values.median()
		Lmedian.append(median)

		# Third Quartile
		third_qrt = feature_values.quantile(0.75)
		Lthird_qrt.append(third_qrt)

		# Maximum
		max_value = feature_values.max()
		Lmax_value.append(max_value)

		# Standard Deviation
		stand_dev = feature_values.std()
		Lstand_dev.append(stand_dev)

	# Add all lists to a list
	# This will be used for creating the dataframe
	All_Cont_vals.extend([
	        count, Lperc_missing, Lcard,
	        Lmin_value, Lfirst_qrt, Lmean, Lmedian, 
	        Lthird_qrt, Lmax_value, Lstand_dev
	    ])

	Cont_df = Create_DataFrame(Cont_Features, Cont_Columns, All_Cont_vals)

	# Write the DatFrames to csv files
	Cont_df.to_csv('C16460726CONT.csv', index_label = 'FEATURENAME')
	Cat_df.to_csv('C16460726CAT.csv', index_label = 'FEATURENAME')













	# # For Continous features
	# min_value = 0
	# first_qrt = 0
	# mean = 0
	# median = 0
	# third_qrt = 0
	# max_value = 0
	# stand_dev = 0

	# # For Both
	# count = 0
	# perc_missing = 0
	# count_missing = 0
	# card = 0

	# d = dict(df_input_params.apply(pd.Series.nunique))
	# count = len(df_input_params)

	# for i in Features:
	# 	# Count
	# 	array = df_input_params[i]

	# 	d2 = dict(array.value_counts())
	# 	# count_missing = d[' ?']

	# 	# % Missing
	# 	array2 = set(array)
	# 	count_missing = array.isna().sum()

	# 	if count_missing == 0:
	# 		perc_missing = 0
	# 	else:
	# 		perc_missing = (count_missing / count) * 100

	# 	# Cardinality
	# 	card = d[i]

	# 	# Minimum
	# 	min_value = array.min()

	# 	# First Quartile
	# 	first_qrt = array.quantile(0.25)

	# 	# Mean
	# 	mean = array.mean()

	# 	# Median
	# 	median = array.median()

	# 	# Third Quartile
	# 	third_qrt = array.quantile(0.75)

	# 	# Maximum
	# 	max_value = array.max()

	# 	# Standard Deviation
	# 	stand_dev = array.std()

	# 	# print(array.value_counts())
	# 	# print("\n")
	# 	print(i)
	# 	print("count", count)
	# 	# print("count_missing", count_missing)
	# 	print("perc_missing", perc_missing)
	# 	print("card", card)
	# 	# print("min_value", min_value)
	# 	# print("first_qrt", first_qrt)
	# 	# print("mean", mean)
	# 	# print("median", median)
	# 	# print("third_qrt", third_qrt)
	# 	# print("max_value", max_value)
	# 	# print("stand_dev", stand_dev)
	# 	# print("mode:",mode)
	# 	# print("\n")
	# 	# print("mode_freq:",mode_freq)
	# 	# print("\n")
	# 	# print("mode_perc:",mode_perc)
	# 	# print("\n")
	# 	# print("mode2:",mode2)
	# 	# print("\n")
	# 	# print("mode2_freq:",mode2_freq)
	# 	# print("\n")
	# 	# print("mode2_perc:",mode2_perc)
	# 	print("\nNEXT\n")


def main():
	process_dir()

main()