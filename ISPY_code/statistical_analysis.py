import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency as _chi2
from scipy import stats
import numpy as np

def categorical_vars(df):

    """
    Realize bar plots showing normalized distribution of categorical variables.
    Arguments:
        df: dataframe. Data with the values to be plotted
    Returns:
        Dataframe with the renamed data.
    """

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
    df.race_id = df.race_id.replace([1,3,4,5,6,50,0],['Caucasian', 
                                                    'African American', 
                                                    'Asian', 
                                                    'Native Hawaiian', 
                                                    'American Indian', 
                                                    'Multiple race', 'Unknown'])

    # Plots for categorical data
    categorical_feats = categorical_vars + [ 'RFS_code', 'RCB', 'Survival', 
                                            'Laterality', 'HR_HER2', 
                                            'HR_HER2_STATUS', 'race_id']
    fig, axs = plt.subplots(nrows=4, ncols=4)
    for feat, ax in zip(categorical_feats, axs.ravel()):
        df[feat].value_counts(normalize = True).sort_index().plot(kind='bar', ax=ax, title=feat, rot=0, figsize=(15,17))

    # Rotate tick labels of some plots
    axs[2,-1].set_xticklabels(axs[2,-1].get_xticklabels(),  rotation=45, ha='right', rotation_mode='anchor')
    axs[3,0].set_xticklabels(axs[3,0].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Do not show axis without plots    
    for ax in axs[3,1:4]:
        ax.set_visible(False)
        
    # Figure title    
    fig.suptitle('Normalized distribution of categorical values', fontsize=16)

    return df


def numerical_vars(df):

    """
    Realize density plots showing distribution of numerical variables.
    Arguments:
        df: dataframe. Data with the values to be plotted
    Returns:
        Nothing.
    """

    # Plots for numerical data
    numeric_feats = [ 'age', 'MRI_LD_T1', 'MRI_LD_T2', 'MRI_LD_T3', 
                        'MRI_LD_T4', 'Survival_length', 'RFS']
    fig, axs = plt.subplots(nrows=4, ncols=2)
    for feat, ax in zip(numeric_feats, axs.ravel()):
        df[feat].plot(kind='density',  ax=ax, title=feat, figsize=(15,19))

    # Do not show axis without plots    
    axs[3, 1].set_visible(False)

    # Figure title    
    fig.suptitle('Density plots of numerical variables', fontsize=16)



def corr_categorical(predictors, output, data, alpha=0.05):

    """
    Performs the chi square test for an output variable and all its categorical 
    predictors to check if the variables are correlated. For those pairs of 
    variables where the null hypothesis is rejected (e.g. they are correlated), 
    bar plots are created showing the output variable distribution by the 
    predictor values.
    Arguments:
        predictors: list of strings. List of all the categorical predictors to
            perform chi squate test against output variable.
        output: string. Categorical variable to perform chi square test
            against predictors. 
        data: dataframe. Data with the variables
        alpha: float. Used to define the confidence level of the test. 
            By default, setted to 0.05 to define a confidence level of 95%.
    Returns:
        df_chi: dataframe. Results of the chi square tests performed. Includes
            the chi square value, the p-value and the conclusion (e.g. whether 
            if Null hypothesis has been rejected or not) for every tests.
    """

    # Dataframe that will store results
    df_chi = pd.DataFrame(np.zeros((len(predictors), 3)))
    df_chi = df_chi.set_index([predictors])
    df_chi.columns = ['chi_square', 'p-value', 'conclusion']
    
    vars_H1 = []
    
    fig = plt.figure()
    
    for i, var in enumerate(predictors):
        
        # create contingency table
        contigency = pd.crosstab(data[var], data[output], margins=True, margins_name="Total")
        # Calculation of Chisquare
        chi2, p , _, _= _chi2( contigency.values )
        # Conclusion
        conclusion = "Failed to reject the null hypothesis."
        if p <= alpha:
            conclusion = "Null Hypothesis is rejected."
            vars_H1.append(var)

        # Fill results dataframe
        df_chi.iloc[i, 0] = chi2
        df_chi.iloc[i, 1] = p
        df_chi.iloc[i, 2] = conclusion
        
    # Print plots of variables with null hypothesis rejected
    if len(vars_H1) == 7:
        r = 3
        fsize = (13,17) 
    else:
        r=1
        fsize = (10,5)

    fig, axs = plt.subplots(nrows=r, ncols=3, figsize=fsize)
    
    for var, ax in zip(vars_H1, axs.ravel()):
        contig = pd.crosstab(data[output], data[var])
        title = output + ' distribution'
        contig.sort_index().plot(kind='bar', ax=ax, stacked=True, rot=0, 
                                xlabel='', title=title)
    
    # Do not show axis without plots    
    if len(vars_H1) == 7:
        for ax in axs[2,1:3]:
            ax.set_visible(False)
    
    plt.show()
    
    return df_chi


def numeric_var_by_category(df, numeric_feats, categorical_feat):
    
    """
    Realize density plots showing distribution of numerical variables by each
    category of the variables.
    Arguments:
        df: dataframe. Data with the variables.
        numeric_feats: list of strings. List of numerical variables to plot.
        categorical_feat: strings. Categorical feature to plot the densities.
    Returns:
        Nothing.
    """

    # Plots for numerical data 
    fig, axs = plt.subplots(nrows=3, ncols=3)
    for feat, ax in zip(numeric_feats, axs.ravel()):
        for value in df[categorical_feat].unique():
            try:
                df.loc[df[categorical_feat]==value, feat].plot(kind='density', ax=ax, 
                                                                title=feat, figsize=(15,19))
            except ValueError: 
                pass            
        ax.legend(list(df[categorical_feat].unique()))
    
    # Do not show axis without plots    
    axs[2, 1].set_visible(False)
    axs[2, -1].set_visible(False)

    # Figure title    
    title= 'Density distribution of numerical variables by ' + categorical_feat + ' categories'
    fig.suptitle(title, fontsize=16)
    

def rfs_by_category(df, categorical_feats):
    
    """
    Realize density plots showing distribution of numerical variables by each
    category of the variables.
    Arguments:
        df: dataframe. Data with the variables.
        numeric_feats: list of strings. List of numerical variables to plot.
        categorical_feat: strings. Categorical feature to plot the densities.
    Returns:
        Nothing.
    """
    
    # Density plot of RFS by categorical values
    fig, axs = plt.subplots(nrows=4, ncols=4, sharey=True, figsize=(12,15))
    for feat, ax in zip(categorical_feats, axs.ravel()):
        df_sorted = df[[feat, 'RFS']].sort_values(by=feat)
        sns.boxplot(ax=ax, x=feat, y='RFS', data=df_sorted).set(xlabel='', ylabel='RFS (days)', title=feat)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=0)

    # Rotate tick labels
    axs[2,2].set_xticklabels(axs[2,-1].get_xticklabels(),  rotation=45, ha='right', rotation_mode='anchor')
    axs[2,3].set_xticklabels(axs[2,-1].get_xticklabels(),  rotation=45, ha='right', rotation_mode='anchor')
    axs[3,0].set_xticklabels(axs[3,0].get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    
    # Do not show axis without plots    
    for ax in axs[3,1:4]:
        ax.set_visible(False)
    
    # Figure title    
    fig.suptitle('RFS boxplots for each categorical variable', fontsize=16)
