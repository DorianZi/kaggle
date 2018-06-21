import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import sys
import seaborn as sns
import matplotlib.pyplot as plt

# https://www.kaggle.com/c/house-prices-advanced-regression-techniques

class housePricePredictor:
    def __init__(self):
        self.train_csv = "train.csv"
        self.test_csv = "test.csv"
  
    def getTrainDF(self):
        self.trainDF = pd.read_csv(self.train_csv)
       
    def dropColumnsTrainDF(self):
        # Street is same
        print "============================ Drop Redundant Columns ====================================="
        self.redundantCols = ["Street","Utilities","Condition2","Heating","LowQualFinSF","GarageYrBlt",\
                              "GarageCars","GarageCond"]
        self.trainDF = self.trainDF.drop(self.redundantCols,axis=1)
        print "Dropped columns:\n", self.redundantCols
        print "=====================================================================================\n"

    def dropTopNaTrainDF(self):
        print "============================ Drop Top Na Columns ====================================="
        count = self.trainDF.isnull().sum().sort_values(ascending=False)
        ratio = count/len(self.trainDF)
        na_data=pd.concat([count,ratio],axis=1,keys=['lost_count','lost_ratio']) 
        #print "Top Na Columns:\n",na_data.head(20)
        cols = na_data['lost_ratio'] > 0.15
        drop_index = na_data[cols].index
        self.trainDF = self.trainDF.drop(drop_index,axis=1)
        print "Dropping whose lost_ratio > 0.15"
        #print "After drop: self.trainDF.shape=",self.trainDF.shape
        print "Dropped columns:\n", drop_index
        print "Remaining columns: \n", self.trainDF.columns    
        print "=====================================================================================\n"
         
    def checkTrainDFcorr(self):
        k=10
        corrmat = self.trainDF.corr()
        cols = corrmat.nlargest(k,'SalePrice')['SalePrice'].index
        cm = np.corrcoef(self.trainDF[cols].values.T)
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()


    def getDummyTrainDF(self):
        self.dm_trainDF = pd.get_dummies(self.trainDF)
        
    def resolveNaTrainDF(self):    
        self.dm_trainDF = self.dm_trainDF.fillna(self.dm_trainDF.mean())
        self.dm_trainDF = self.dm_trainDF.fillna(0)
    
    def getTestDF(self):
        self.testDF = pd.read_csv(self.test_csv)

    def getDummyTestDF(self):
        self.dm_testDF = pd.get_dummies(self.testDF)

    def setTestColumnsAsTrain(self):
        trainCols = self.dm_trainDF.columns.values.tolist()
        self.dm_testDF = self.dm_testDF.loc[:,trainCols]

    def resolveNaTestDF(self):    
        self.dm_testDF = self.dm_testDF.fillna(self.dm_testDF.mean())
        self.dm_testDF = self.dm_testDF.fillna(0)

    def getTrainTestValues(self): 
        self.X_train = self.dm_trainDF.drop(['Id','SalePrice'],axis=1).values
        self.Y_train = self.dm_trainDF['SalePrice'].values
        
        self.X_test_ids = self.dm_testDF['Id'].values
        self.X_test = self.dm_testDF.drop(['Id','SalePrice'],axis=1).values


    def createModel(self):
        print "============================Create Train Model=========================================="
        self.rft = RandomForestRegressor()
        print "======================================================================================\n"

    def train(self):
        print "===================================Train================================================"
        self.rft.fit(self.X_train,self.Y_train)
        print "======================================================================================\n"

    def predict(self):
        print "===================================Predict=============================================="
        self.predict = self.rft.predict(self.X_test)
        print "======================================================================================\n"

    def savePrediction(self,filename):    
        print "================================Save Prediction========================================="
        with open(filename,'w') as f:
            f.write("Id,SalePrice\n")
            for i in range(len(self.predict)):
                f.write("{0},{1}\n".format(self.X_test_ids[i],self.predict[i]))
        print "======================================================================================\n"


    
    
 
if __name__ == "__main__":
    predictor = housePricePredictor()
    predictor.getTrainDF()
    predictor.dropColumnsTrainDF()
    predictor.checkTrainDFcorr()
    predictor.dropTopNaTrainDF()
    predictor.getDummyTrainDF()
    predictor.resolveNaTrainDF()

    predictor.getTestDF()
    predictor.getDummyTestDF()
    predictor.setTestColumnsAsTrain()
    predictor.resolveNaTestDF()

    predictor.getTrainTestValues()
    predictor.createModel()
    predictor.train()
    predictor.predict()
    predictor.savePrediction("results.csv")

