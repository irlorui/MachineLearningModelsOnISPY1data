import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def categorical_vars_analysis(df):
    
    # Rename clinical outcomes and predictors 0/1 to make it easier to display
    categorical_vars = ['ER+','PR+','HR+', 'HER2+', 'Bilateral','PCR']
    for str_ in categorical_vars:
        df[str_] = df[str_].replace([1,0],['Yes','No'])
        
    # rename other predictors and outcomes
    df.Survival = df.Survival.replace([7,8,9], ['Yes','No','Lost'])
    df.Laterality = df.Laterality.replace([1,2],['Left','Right'])
    df.RFS_code = df.RFS_code.replace([0,1],['No','Yes'])
    df.RCB = df.RCB.replace([0,1,2,3],['0','I', 'II', 'III'])
    df['HR_HER2'] = df['HR_HER2'].replace([1,2,3],['HR+ Her2-', 'Her2+','Triple Negative'])
    df.race_id = df.race_id.replace([1,3,4,5,6,50,0],['Caucasian', 'African American', 'Asian', 
                                                    'Native Hawaiian', 
                                                    'American Indian', 
                                                    'Multiple race', 'Unknown'])

    # Plots for categorical data
    categorical_feats = categorical_vars + [ 'RFS_code', 'RCB', 'Survival', 'Laterality', 'HR_HER2', 'HR_HER2_STATUS', 'race_id']

    fig, axs = plt.subplots(nrows=4, ncols=4)

    for feat, ax in zip(categorical_feats, axs.ravel()):
        df[feat].value_counts(normalize = True).sort_index().plot(kind='bar', ax=ax, title=feat, rot=0, figsize=(15,17))

    axs[2,-1].set_xticklabels(axs[2,-1].get_xticklabels(),  rotation=45, ha='right', rotation_mode='anchor')
    axs[3,0].set_xticklabels(axs[3,0].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Do not show axis without plots    
    for ax in axs[3,1:4]:
        ax.set_visible(False)
        
    fig.suptitle('Normalized distribution of categorical values', fontsize=16)

    return df

