import os
import csv
import pandas as pd

def process():
	pathHeart = "../Data/heart-disease-uci/"

	# Read in dataset
	heart = pd.read_csv(pathHeart + 'heart.csv')

	# Read in all the original datasets
	cleveland = pd.read_csv(pathHeart + 'cleveland.txt', header=None).replace(' ', ',', regex=True)
	hungarian = pd.read_csv(pathHeart + 'hungarian.txt', header=None).replace(' ', ',', regex=True)
	switzerland = pd.read_csv(pathHeart + 'switzerland.txt', header=None).replace(' ', ',', regex=True)

	# Replace all spaces with a comma
	# print(cleveland.replace(' ', ',', regex=True))

	print(cleveland)

	# for _, row in cleveland.iterrows():
	# 	print(row[0])

process()