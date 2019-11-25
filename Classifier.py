# Start by importing libraries
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def train_model():
	# Load heart disease dataset into pandas dataframe
	pathHeart = "../Data/heart-disease-uci/"
	heart = pd.read_csv(pathHeart + 'heart.csv')

	print(heart.describe())

train_model()