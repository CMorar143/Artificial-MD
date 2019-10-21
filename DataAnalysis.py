import os
import csv
import pandas as pd


def process_dir():
    path = "../Data/Health_Survey/"
    flist = os.listdir(path)
    featureNames = []

    demographic = pd.read_csv(path + 'demographic.csv')
    diet = pd.read_csv(path + 'diet.csv')
    examination = pd.read_csv(path + 'examination.csv')
    labs = pd.read_csv(path + 'labs.csv')
    medications = pd.read_csv(path + 'medications.csv', encoding = "ISO-8859-1")
    questionnaire = pd.read_csv(path + 'questionnaire.csv')
    # print(medications.head())

    featureNames = pd.read_csv(path + './Feature_Dictionary/FeatureNames.txt')
    # print(featureNames.head())
    # fname = list(featureNames)
    # print("fname length:", len(fname))
    desc = pd.read_csv(path + './Feature_Dictionary/NAHNES_2014_Dictionary.csv')
    # print(desc.head())
    
    d = dict(zip(list(desc['Variable Name']), list(desc['Variable Description'])))
    # print(list(d.keys())[list(d.values()).index('Respondent sequence number')])

    list_of_keys = list(d.keys())

    # print(len(list_of_keys))
    print()
    
    # For checking cardinality
    d2 = dict(demographic.apply(pd.Series.nunique))

    # demographic.fillna("?", inplace=True)

    c = 0
    # print(demographic)
    # print("\n", list_of_keys)
    for i in demographic:
        # count = demographic[i].isna().sum()
        # print("\n")
        # print(demographic[i].describe())
        # print(list_of_keys[c], d[list_of_keys[c]])

        c = c + 1
    print()
    print(c)


    # # For Categorical features
    # mode = ''
    # mode_freq = 0
    # mode_perc = 0
    # mode2 = ''
    # mode2_freq = 0
    # mode2_perc = 0

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

    # d = dict(demographic.apply(pd.Series.nunique))
    # count = demographic['id,'].count()

    # for i in Cat_Features:
    #     # Count
    #     array = demographic[i + ',']

    #     d2 = dict(array.value_counts())
    #     # count_missing = d[' ?']
        
    #     # % Missing
    #     array2 = set(array)
    #     if (' ?') in array2:
    #         count_missing = d2[' ?']
    #         perc_missing = (count_missing / count) * 100
    #     else:
    #         count_missing = 0
    #         perc_missing = 0

    #     # Cardinality
    #     card = d[i + ',']

    #     # if count_missing > 0:
    #         # card = card - 1

    #     # Mode
    #     mode = list(d2)[0]

    #     # Mode Freq.
    #     mode_freq = d2[mode] #array.value_counts()[array.mode()]

    #     # Mode %
    #     mode_perc = (mode_freq / count) * 100

    #     # 2nd Mode
    #     mode2 = list(d2)[1]

    #     # 2nd Mode Freq.
    #     mode2_freq = d2[mode2]

    #     # 2nd Mode %
    #     mode2_perc = (mode2_freq / count) * 100

    #     print(array.describe())
    #     print("\n")
    #     print(array.value_counts())
    #     print("\n")
    #     print("count:", count)
    #     print("\n")
    #     print("count_missing:",count_missing)
    #     print("\n")
    #     print("perc_missing:",perc_missing)
    #     print("\n")
    #     print("card:",card)
    #     print("\n")
    #     print("mode:",mode)
    #     print("\n")
    #     print("mode_freq:",mode_freq)
    #     print("\n")
    #     print("mode_perc:",mode_perc)
    #     print("\n")
    #     print("mode2:",mode2)
    #     print("\n")
    #     print("mode2_freq:",mode2_freq)
    #     print("\n")
    #     print("mode2_perc:",mode2_perc)
    #     print("\n\n\n\nNEXT\n\n\n\n")


    # for i in Cont_Features:
    #     # Count
    #     array = demographic[i + ',']

    #     d2 = dict(array.value_counts())
    #     # count_missing = d[' ?']
            
    #     # % Missing
    #     array2 = set(array)
    #     if (' ?') in array2:
    #         count_missing = array.value_counts()[' ?']
    #         perc_missing = (count_missing / count) * 100
    #     else:
    #         count_missing = 0
    #         perc_missing = 0

    #     # Cardinality
    #     card = d[i + ',']

    #     # Minimum
    #     min_value = array.min()

    #     # First Quartile
    #     first_qrt = array.quantile(0.25)

    #     # Mean
    #     mean = array.mean()

    #     # Median
    #     median = array.median()

    #     # Third Quartile
    #     third_qrt = array.quantile(0.75)

    #     # Maximum
    #     max_value = array.max()

    #     # Standard Deviation
    #     stand_dev = array.std()

        # print(array.value_counts())
        # print("\n")
        # print(count)
        # print("\n")
        # print(count_missing)
        # print("\n")
        # print(perc_missing)
        # print("\n")
        # print(card)
        # print("\n")
        # print(min_value)
        # print("\n")
        # print(first_qrt)
        # print("\n")
        # print(mean)
        # print("\n")
        # print(median)
        # print("\n")
        # print(third_qrt)
        # print("\n")
        # print(max_value)
        # print("\n")
        # print(stand_dev)
        # print("\n\n\n\nNEXT\n\n\n\n")


    # Cont = demographic.to_csv('C16460726CONT.csv', index_label = 'FEATURENAME')
    # Cat = demographic.to_csv('C16460726CAT.csv', index_label = 'FEATURENAME')



def main():
    process_dir()

main()