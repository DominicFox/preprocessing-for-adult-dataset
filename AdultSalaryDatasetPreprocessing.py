import pandas as pd
import os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score

from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def ReplaceQsMarkWithNaN (data):
    return data.replace(to_replace = ' ?', value = np.nan)

def GetPercentageOfNaNValuesPerColumn (data):
    totalCount = data.count()
    NaNCount = data.isnull().sum()
    return 100*NaNCount / (totalCount + NaNCount)

def GetCategoricalColumns (data):
    categoricalColumns = (data.dtypes == 'object')
    categoricalColumns = list(categoricalColumns[categoricalColumns].index)
    return categoricalColumns

def GetNumericalColumns(data):
    numColumns = (data.dtypes == 'int64')
    numColumns = list(numColumns[numColumns].index)
    return numColumns

def LoadDataset(filepathTrain, filepathTest):
    # Read the data
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race','sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'salary']
    datasetTrain = pd.read_csv(filepathTrain, names = columns)
    datasetTest = pd.read_csv(filepathTest, names = columns)

    # Replace ? with NaN
    datasetTrain = ReplaceQsMarkWithNaN(datasetTrain)
    datasetTest = ReplaceQsMarkWithNaN(datasetTest)

    datasetTrain.drop('education-num', axis = 1, inplace=True)
    datasetTest.drop('education-num', axis = 1, inplace=True)

    # Drop first row of test set
    datasetTest.drop(datasetTest.index[0], inplace=True)

    # Split target feature away from the others
    XTrain = datasetTrain.drop('salary', axis = 1)
    yTrain = datasetTrain['salary']

    XTest = datasetTest.drop('salary', axis = 1)
    yTest = datasetTest['salary']

    # Remove full-stops in 'yTest'
    yTest = yTest.replace(to_replace={' <=50K.': ' <=50K', ' >50K.':' >50K'})

    return XTrain, yTrain, XTest, yTest

def CreateModels():
    models = []
    models.append(['RidgeClassifier', RidgeClassifier()])
    models.append(['LogisticRegression', LogisticRegression(solver='liblinear')])
    models.append(['SGD', SGDClassifier()])
    models.append(['KNN', KNeighborsClassifier()])
    models.append(['LDA', LinearDiscriminantAnalysis()])
    models.append(['GaussianNB', GaussianNB()])
    models.append(['Tree', DecisionTreeClassifier()])
    models.append(['XGB', XGBClassifier()])
    return models

if __name__ == '__main__':
    # Load Dataset
    filepathTrain = os.path.join(os.getcwd(), "Data/adult.data")
    filepathTest = os.path.join(os.getcwd(), "Data/adult.test")
    
    XTrain, yTrain, XTest, yTest = LoadDataset(filepathTrain, filepathTest)

    # Impute missing values
    imputer = SimpleImputer(strategy='most_frequent')
    XTrainImputed = pd.DataFrame(imputer.fit_transform(XTrain))
    XTestImputed = pd.DataFrame(imputer.transform(XTest))

    # Imputing removed column names; add them back
    XTrainImputed.columns = XTrain.columns
    XTestImputed.columns = XTest.columns

    # Imputing has made data type of all columns 'object'; reinstate the old data types
    XTrainImputed = pd.DataFrame(XTrainImputed, columns=XTrain.columns).astype(XTrain.dtypes.to_dict())
    XTestImputed = pd.DataFrame(XTestImputed, columns=XTest.columns).astype(XTest.dtypes.to_dict())

    # Ordinal encode the education column
    ordinalEncodingOrder = [' Preschool', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th', ' 11th', ' 12th', ' HS-grad',' Some-college', ' Assoc-voc', ' Assoc-acdm', ' Bachelors', ' Masters', ' Prof-school', ' Doctorate']
    ordinalEncoder = OrdinalEncoder(categories=[ordinalEncodingOrder])

    XTrainEncoded = XTrainImputed.copy()
    XTestEncoded = XTestImputed.copy()

    XTrainEncoded['education'] = ordinalEncoder.fit_transform(XTrainImputed.loc[:,['education']])
    XTestEncoded['education'] = ordinalEncoder.transform(XTestImputed.loc[:,['education']])

    # One-Hot Encode the columns
    OHEncoder = OneHotEncoder(sparse=False)
    onehotColumns = GetCategoricalColumns(XTrainEncoded)

    # Create new dataframe to hold OH encoded columns
    XTrainOHCols = pd.DataFrame(OHEncoder.fit_transform(XTrainEncoded[onehotColumns]))
    XTestOHCols = pd.DataFrame(OHEncoder.transform(XTestEncoded[onehotColumns]))
    
    # Put the index back in
    XTrainOHCols.index = XTrain.index
    XTestOHCols.index = XTest.index

    # Extract out the numerical columns
    XTrainNumericalCols = XTrainEncoded.drop(onehotColumns, axis = 1)
    XTestNumericalCols = XTestEncoded.drop(onehotColumns, axis = 1)
    
    # Concatenate the OH columns and the numerical columns to form complete encoded set
    XTrainEncoded = pd.concat([XTrainOHCols, XTrainNumericalCols], axis = 1)
    XTestEncoded = pd.concat([XTestOHCols.reset_index(drop=True), XTestNumericalCols.reset_index(drop=True)], axis = 1)

    # Set dtype of column names to string
    XTrainEncoded.columns = XTrainEncoded.columns.astype(str)
    XTestEncoded.columns = XTestEncoded.columns.astype(str)

    # Scale numerical columns
    scaler = MinMaxScaler()
    numericColumns = GetNumericalColumns(XTrain)

    XTrainEncoded[numericColumns] = scaler.fit_transform(XTrainEncoded[numericColumns])
    XTestEncoded[numericColumns] = scaler.transform(XTestEncoded[numericColumns])

    # Encode the target variable
    labelEnc = LabelEncoder()

    yTrainEncoded = pd.Series(labelEnc.fit_transform(yTrain))
    yTestEncoded = pd.Series(labelEnc.transform(yTest))

    yTrainEncoded.index = yTrain.index
    yTestEncoded.index = yTest.index

    models = CreateModels()

    for name, model in models:
        model.fit(XTrainEncoded, yTrainEncoded)
        predictions = model.predict(XTestEncoded)
        print('%s: %.3f' % (name, accuracy_score(yTestEncoded, predictions)))

    

