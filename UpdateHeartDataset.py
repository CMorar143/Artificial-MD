import os
import csv
import pandas as pd

def process():
	pathHeart = "../Data/heart-disease-uci/"

	# Read in dataset
	heart = pd.read_csv(pathHeart + 'heart.csv')

	# Read in all the original datasets
	cleveland = pd.read_csv(pathHeart + 'cleveland.txt', header=None)
	hungarian = pd.read_csv(pathHeart + 'hungarian.txt', header=None)
	switzerland = pd.read_csv(pathHeart + 'switzerland.txt', header=None)

	
	# for _, row in cleveland.iterrows():
	# 	print(row[0])

process()