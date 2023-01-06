import pandas as pd


def load_data(filepath):

    # Load data
    #file = './data/I-SPY 1 All Patient Clinical and Outcome Data.xlsx'
    sheet_predictors = 'TCIA Patient Clinical Subset'
    sheet_outcomes = 'TCIA Outcomes Subset'

    # load and set index of predictors and drop Columns with metadata
    predictors = pd.read_excel(filepath, sheet_name=sheet_predictors)
    predictors = predictors.set_index('SUBJECTID')
    predictors.drop(['DataExtractDt'],axis=1,inplace=True)

    # load and set index of outcomes and drop Columns with metadata
    outcomes_df = pd.read_excel(filepath, sheet_name=sheet_outcomes)
    outcomes_df.drop(['DataExtractDt'],axis=1,inplace=True)
    outcomes_df = outcomes_df.set_index('SUBJECTID')

    #merge outcomes and predictors using the Subject ID index
    ISPY = predictors.join(outcomes_df)

    #Rename columns
    ISPY = ISPY.rename(columns={'ERpos':'ER+',
                            'PgRpos':'PR+',
                            'HR Pos':'HR+',
                            'Her2MostPos': 'HER2+',
                            'HR_HER2_CATEGORY': 'HR_HER2',
                            'BilateralCa':"Bilateral",
                            'sstat':'Survival',
                            'MRI LD Baseline':'MRI_LD_T1',
                            'MRI LD 1-3dAC':'MRI_LD_T2',
                            'MRI LD InterReg':'MRI_LD_T3',
                            'MRI LD PreSurg': 'MRI_LD_T4',
                            'survDtD2 (tx)':'Survival_length',
                            'rfs_ind':'RFS_code',
                            'RCBClass':'RCB'})


    return ISPY