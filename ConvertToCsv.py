import os
import csv
import pandas as pd



path = "../Data/Health_Survey/"
pathHeart = "../Data/heart-disease-uci/"
flist = os.listdir(path)
featureNames = []