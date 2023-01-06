import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import ADASYN 
import math

RANDOM_STATE = 14 

def split_data(Xdata,Ydata, oversampling=False, k=5, stratify=True):
    
    """
    Split data into train and test datasets.
    Arguments:
        Xdata: dataframe. Predictor variables.
        Ydata: dataframe. Output variable with data labels.
        oversampling: boolean. Specify if minority class should be oversampled.
            By default, set to False.
        k: integer. Number of nearest neighbors to use for oversampling data.
        stratify: boolean. Specify if data split should be stratify to have same
            number of each class on train and test data. By default, set to True.
    Returns:
        X_train, X_test, y_train, y_test. Dataframes with the train and test data
            for the predictors and the output variable
    """

    if stratify == True:
        X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata,
                                                            test_size=0.33, 
                                                            stratify=Ydata,
                                                            random_state=RANDOM_STATE)
    else:
        X_train, X_test, y_train, y_test = train_test_split(Xdata, Ydata,
                                                            test_size=0.33, 
                                                            random_state=RANDOM_STATE)
    
    if oversampling == True:
        
        print('Oversampling train data with ADASYN method and k=' + str(k) + ' nearest neighbors\n')
        sm = ADASYN(random_state = RANDOM_STATE, n_neighbors = k)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        
        
    return X_train, X_test, y_train, y_test


def binary_classifier_metrics(classifier, Xtest, Ytest, classes):
    
    """
    Performs metrics for binary classifier. AUC, False-positive rate, 
    True-positive rate, classification report are computed and printed. 
    Confusion matrix is also calculated and plotted.
    Arguments:
        classifier: Trained model to create the metrics with test data.
        Xtest: dataframe. Test data predictor variables.
        Ytest: dataframe. Test data output variable.
        classes: list of strings. Specifies the labels of the output classes.
    Returns:
        auc: float. AUC computed for the classifier
        fpr, tpr: array of numbers. False-positive and True-positive rate.
    """

    predicted_class = classifier.predict(Xtest)
    
    # metrics
    auc = metrics.roc_auc_score(Ytest,predicted_class)
    auc = np.round(auc,3)

    # ROC curve
    probability = classifier.predict_proba(Xtest)
    fpr, tpr, _  = metrics.roc_curve(Ytest, probability[:,1], 
                                     drop_intermediate=False)

    # report
    print(metrics.classification_report(Ytest, predicted_class, labels=[0,1], 
                                        target_names=classes))
    print('The estimated AUC is ' + str(auc))
    print('\n'*2)
    
    #Confusion matrix
    conf_matrix = confusion_matrix(Ytest, predicted_class)
    fig,axis = plt.subplots(figsize=(5, 4))
    sns.heatmap(conf_matrix, cmap="Blues" ,annot=True, fmt= '.0f',ax=axis)
    axis.set_title("Confusion matrix")
    axis.set_xlabel('Predicted')
    axis.set_xticklabels(classes)
    axis.set_ylabel('Real')
    axis.set_yticklabels(classes)

    return auc, fpr, tpr



def plot_roc(auc1_, fpr1_, tpr1_, auc2_, fpr2_, tpr2_, title ='ROC curve'):
    
    plt.figure(figsize=(10, 5))
    plt.plot(fpr1_, tpr1_, fpr2_, tpr2_);
    plt.legend(['AUC = ' + str(auc1_)]);
    plt.xlabel('False-positive rate');
    plt.ylabel('True-positive rate');
    plt.legend(['Unbalanced AUC = ' + str(auc1_),'Oversampled AUC = ' + str(auc2_)]);
    plt.title(title);
    
    
def plot_feature_importance(clf_initials, classifier, predictors):
    
    """
    Generates bar plot to show the importance of each variable of a model.
    Arguments:
        clf_initials: string. Initials to distinguish model type.
        classifier: dataframe. Trained model to evaluate variable importance.
        predictors: list of strings. Variables names to use as tick labels. 
    Returns:
        Nothing
    """
    # Feature importance
    fig,axis = plt.subplots(figsize=(5, 4))
    
    # For SVM models, feature importance is stored in coef_ attribute
    if clf_initials == 'SVM':
        feat_imp = pd.Series(abs(classifier.coef_[0]), index=predictors).nlargest(10).sort_index()
    else:
        feat_imp = pd.Series(classifier.feature_importances_, predictors).sort_index()
    
    feat_imp.plot(kind='barh', title=('Feature Importances - ' + clf_initials + ' model'))
    plt.xlabel('Feature Importance Score')


def train_test_classifier(Xdata, Ydata, clf, clf_name, pars, score, outcome, 
                         classes=['NO', 'YES'], oversampling=False, k=5, cv=3):
    """
    Creates a classifier from scratch. Using other functions, data is splitted
    into train and test datasets and search for the hyperparameters using a grid
    search with specified parameters. Model with best hyperparameters is trained
    and then evaluated with test data using binary_classifier_metrics function.
    Arguments:
        Xdata: dataframe. Predictor variables.
        Ydata: dataframe. Output variable with data labels.
        clf: classifier. Trained model to create the metrics with test data.
        clf_name: string. Model name.
        pars: dictionary. Parameters used for gird search.
        score: string. Score to apply for the grid search.
        outcome: string. Name of the variable to predict
        classes: list of strings. Specifies the labels of the output classes.
        oversampling: boolean. Specify if minority class should be oversampled.
            By default, set to False.
        k: integer. Number of nearest neighbors to use for oversampling data.
        cv: integer. number of folds to perform the cross-validation
    Returns:
        float. AUC computed for the classifier
        fpr, tpr: array of numbers. False-positive and True-positive rate.
        grid.best_estimator_: classifier. Best classifier obtained in the
            hyperparameter tuning
    """
    # Basic info about model
    print('=='*30)
    print(clf_name + ' predicting ' + outcome)
    print('=='*30)   
    
    # split data
    X_train, X_test, y_train, y_test = split_data(Xdata, Ydata, oversampling, k)

    # perform grid search for hyperparameter tuning
    grid=  GridSearchCV(clf, param_grid = pars, scoring = score, cv= cv, 
                        verbose = 0, n_jobs = -1)

    # fit
    grid.fit(X_train,y_train)
    
    # Print best params of model
    print('Best parameters: ')
    print(grid.best_params_)
    print('--'*30 + '\n')

    # metrics
    auc, fpr, tpr = binary_classifier_metrics(grid.best_estimator_, 
                                              X_test, y_test, classes)
    plt.show()
       
    # output
    return auc, fpr, tpr, grid.best_estimator_


def regressor_metrics(regressor, X_test, y_test):
    
    """
    Performs metrics for regressor. Median absolute error, mean absolute error
    and root mean squared error are printed and returned.
    Arguments:
        regressor: regressor. Trained model to create the metrics with test data.
        X_test: dataframe. Test data predictor variables.
        y_test: dataframe. Test data output variable.
    Returns:
        y_pred, y_test: numeric arrays. Contains predicted and real data of
            dependent variable. 
        median_ae: float. Median absolute error of model predictions.
        mae: float. Mean absolute error of model predictions on test data.
        rmae: float. Root mean squared error of model predictions on test data.
    """

    y_pred = regressor.predict(X_test)
    
    # Median absolute error
    median_ae = metrics.median_absolute_error(y_test, y_pred)
    median_ae = np.round(median_ae, decimals=3)
    
    # Mean absolute error
    mae = metrics.mean_absolute_error(y_test, y_pred)
    mae = np.round(mae, decimals=3)
    
    # Root mean squared error
    rmae = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
    rmae = np.round(rmae, decimals=3)
    
    
    # report
    print('Median absolute error: ' + str(median_ae))
    print('Mean absolute error: ' + str(mae))
    print('Root mean squared error: ' + str(rmae))
    print('\n'*2)
    
    return y_test, y_pred, median_ae, mae, rmae


def train_test_regressor(Xdata, Ydata, regressor, reg_name, pars, score, outcome, 
                         oversampling=False, k=5, cv=3):
    """
    Creates a regressor from scratch. Using other functions, data is splitted
    into train and test datasets and search for the hyperparameters using a grid
    search with specified parameters. Model with best hyperparameters is trained
    and then evaluated with test data using regressor_metrics function.
    Arguments:
        Xdata: dataframe. Predictor variables.
        Ydata: dataframe. Output variable with data labels.
        reg_name: string. Model name.
        pars: dictionary. Parameters used for gird search.
        score: string. Score to apply for the grid search.
        outcome: string. Name of the variable to predict
        oversampling: boolean. Specify if minority class should be oversampled.
            By default, set to False.
        k: integer. Number of nearest neighbors to use for oversampling data.
        cv: integer. number of folds to perform the cross-validation
    Returns:
        grid.best_estimator_: classifier. Best classifier obtained in the
            hyperparameter tuning.
        y_pred, y_test: numeric arrays. Contains predicted and real data of
            dependent variable. 
        mae: float. Mean absolute error of model predictions on test data.
        rmae: float. Root mean squared error of model predictions on test data.
    """
    # Basic info about model
    print('=='*30)
    print(reg_name + ' predicting ' + outcome)
    print('=='*30)   
    
    # split data
    X_train, X_test, y_train, y_test = split_data(Xdata, Ydata, oversampling, k, stratify=False)

    # perform grid search
    grid=  GridSearchCV(regressor, param_grid = pars, scoring = score, cv= cv, verbose = 0, n_jobs = -1)

    # fit
    grid.fit(X_train,y_train)
    
    
    # Print best params of model
    print('Best parameters: ')
    print(grid.best_params_)
    print('--'*30 + '\n')

    
    # metrics
    y_test, y_pred, median_ae, mae, rmae = regressor_metrics(grid.best_estimator_, X_test, y_test)
    
    # Plot predicted vs real
    fig, ax = plt.subplots(nrows=1, ncols= 1, figsize=(5,5))
    ax = sns.regplot(x = y_test, y= y_pred);
    ax.set_ylabel('Predicted ' + outcome);
    ax.set_xlabel('Observed ' + outcome);
    
           
    # output
    return grid.best_estimator_, y_pred, y_test, mae, rmae

