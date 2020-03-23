import numpy as np
import pandas as pd

def load_dataframe():
    # Load heart disease dataset into pandas dataframe
    pathHeart = "../../FYP_Data/heart-disease-uci/"
    heart = pd.read_csv(pathHeart + 'new_cleveland.csv')
    heart = heart.drop(['dm'], axis=1)
    print(heart.head())
    return heart

heart = load_dataframe()

print(heart)

