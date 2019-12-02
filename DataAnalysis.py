import os
import csv
import pandas as pd


def process_dir():
    path = "../Data/Health_Survey/"
    pathHeart = "../Data/heart-disease-uci/"
    flist = os.listdir(path)
    featureNames = []

    # demographic = pd.read_csv(path + 'demographic.csv')
    diet = pd.read_csv(path + 'diet.csv')
    examination = pd.read_csv(path + 'examination.csv')
    labs = pd.read_csv(path + 'labs.csv')
    # medications = pd.read_csv(path + 'medications.csv', encoding = "ISO-8859-1")
    questionnaire = pd.read_csv(path + 'questionnaire.csv')
    glucose = pd.read_csv(path + 'GLU_H.csv')
    heart = pd.read_csv(pathHeart + 'heart.csv')
    # print(demographic.head())
    # print(diet.head())
    # print(examination.head())
    # print(labs.head())
    # print(medications.head())
    # print(questionnaire.head())

    # dict_combined = {
    #     'Demographic': demographic,
    #     'Diet': diet,
    #     'Examination': examination,
    #     'Labs': labs,
    #     'Medications': medications,
    #     'Questionnaire': questionnaire
    # }

    # df_combined = pd.concat(dict_combined, sort=False)

    df_list = [
        diet,
        examination,
        labs,
        glucose,
        questionnaire
    ]

    input_params = [
        'DIQ010',
        'DIQ160',
        'CDQ010',
        'CDQ001',
        'BPQ080',
        'BPQ020',
        'BMXHT',
        'BMXWT',
        'BMXBMI',
        'BPXCHR',
        'BPXPULS',
        'BPXPTY',
        'BPXSY1',
        'BPXDI1',
        'BPXSY2',
        'BPXDI2',
        'LBXSTP',
        'LBDHDD',
        'LBDLDL',
        'LBXTC',
        'LBXGLU',
        'LBXTR',
        'LBXSUA'
    ]

    df_combined = pd.DataFrame.copy(diet)

    for f in df_list[1:]:
        df_combined = pd.merge(df_combined, f, on='SEQN', sort=False)

    df_combined.set_index('SEQN', inplace=True)

    df_input_params = df_combined[input_params]
    df_input_params.dropna(thresh=10, inplace=True)

    print(df_input_params.head())
    print(df_input_params.tail())

    # seqn_array = df_combined['SEQN']
    # print(df_combined[['SDDSRVYR']])
    # print(df_combined.head())

    # featureNames = pd.read_csv(path + './Feature_Dictionary/FeatureNames.txt')
    # # print(featureNames.head())
    # # fname = list(featureNames)
    # # print("fname length:", len(fname))
    # desc = pd.read_csv(path + './Feature_Dictionary/NAHNES_2014_Dictionary.csv')
    # # print(desc.head())
    
    # d = dict(zip(list(desc['Variable Name']), list(desc['Variable Description'])))
    # # print(list(d.keys())[list(d.values()).index('Respondent sequence number')])

    # list_of_keys = list(d.keys())

    # print(medications['RXDRSC1'].value_counts())

    # print(len(list_of_keys))
    print()
    
    # For checking cardinality
    # d2 = dict(demographic.apply(pd.Series.nunique))

    # demographic.fillna("?", inplace=True)





    # # For Categorical features
    # mode = ''
    # mode_freq = 0
    # mode_perc = 0
    # mode2 = ''
    # mode2_freq = 0
    # mode2_perc = 0

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


    for i in input_params:
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
        # print(i, "\n")
        # print("count", count)
        # print("count_missing", count_missing)
        # print("perc_missing", perc_missing)
        # print("card", card)
        # print("min_value", min_value)
        # print("first_qrt", first_qrt)
        # print("mean", mean)
        # print("median", median)
        # print("third_qrt", third_qrt)
        # print("max_value", max_value)
        # print("stand_dev", stand_dev)
        # print("\n\n\n\nNEXT\n\n\n\n")
        # print("\nNEXT\n")


    # Cont = demographic.to_csv('C16460726CONT.csv', index_label = 'FEATURENAME')
    # Cat = demographic.to_csv('C16460726CAT.csv', index_label = 'FEATURENAME')

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


def main():
    process_dir()

main()