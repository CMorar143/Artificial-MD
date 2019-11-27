import os
import csv
import pandas as pd

def process():

	# Read in original dataset
	pathHeart = "../Data/heart-disease-uci/"
	heart = pd.read_csv(pathHeart + 'heart.csv')
	cleveland = pd.read_csv(pathHeart + 'cleveland.txt', header=None)
	print(cleveland)

process()