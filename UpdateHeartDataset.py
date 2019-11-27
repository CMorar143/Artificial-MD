import os
import csv
import pandas as pd

def process_csv():
	pathHeart = "../Data/heart-disease-uci/"

	# Read in dataset
	heart = pd.read_csv(pathHeart + 'heart.csv')

	# Read in all the original datasets
	cleveland = pd.read_csv(pathHeart + 'cleveland.txt', header=None).replace(' ', ',', regex=True)
	hungarian = pd.read_csv(pathHeart + 'hungarian.txt', header=None).replace(' ', ',', regex=True)
	switzerland = pd.read_csv(pathHeart + 'switzerland.txt', header=None).replace(' ', ',', regex=True)

	cleveland = cleveland.replace(r'\s', ' ', regex=True)
	print(cleveland)

	# Iterate through the individual rows
	# for _, row in cleveland.iterrows():
	# 	print(row[0].replace('\n', ','))

	cleveland.close()
	hungarian.close()
	switzerland.close()


def text_file():
	pathHeart = "../Data/heart-disease-uci/"
	cleveland = open(pathHeart + 'cleveland.txt', mode='r')

	# Read all the lines in the text file
	all_lines = cleveland.readlines()

	print(all_lines[0][1])
	
	# Create a new file
	new_cleveland = open(pathHeart + 'new_cleveland.txt', 'w')

	
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
