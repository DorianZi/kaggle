import os
import pandas as pd
import numpy as np
#from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.cross_validation import train_test_split
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm, skew 

class housePricePredictor:
    def __init__(self):
        self.train_csv = "train.csv"
        self.test_csv = "test.csv"
  
    def getTrainTestDF(self):
        self.trainDF = pd.read_csv(self.train_csv)
        self.testDF = pd.read_csv(self.test_csv)
    
    def drawByTrainDF(self):
        #show the overall of SalePrice
        sns.distplot(self.trainDF['SalePrice'])
        plt.show()
        
        #corr
        k=10
        corrmat = self.trainDF.corr()
        cols = corrmat.nlargest(k,'SalePrice').index
        #print self.trainDF[cols].values
        #print '----\n\n\n',self.trainDF[cols].values.T
        cm = np.corrcoef(self.trainDF[cols].values.T)
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
        plt.show()

        #show the relation to SalePrice
        for var in self.trainDF.select_dtypes(include=[np.number]).columns:
            concatDF = self.trainDF[[var,'SalePrice']]
            concatDF.plot.scatter(x=var,y='SalePrice')
            plt.show()

    def dropRowsTrainDF(self):
        print "============================ Drop Train Rows ======================================="
        print "Before drop: ",self.trainDF.shape
        self.trainDF = self.trainDF.drop(self.trainDF[(self.trainDF['GrLivArea']>4000) & (self.trainDF['SalePrice']<300000)].index)
        self.trainDF = self.trainDF.drop(self.trainDF[(self.trainDF['LotFrontage']>300) & (self.trainDF['SalePrice']<300000)].index)
        self.trainDF = self.trainDF.drop(self.trainDF[(self.trainDF['BsmtFinSF1']>5000) & (self.trainDF['SalePrice']<200000)].index)
        self.trainDF = self.trainDF.drop(self.trainDF[(self.trainDF['TotalBsmtSF']>6000) & (self.trainDF['SalePrice']<200000)].index)
        self.trainDF = self.trainDF.drop(self.trainDF[(self.trainDF['1stFlrSF']>4000) & (self.trainDF['SalePrice']<200000)].index)
        print "After drop: ",self.trainDF.shape
        print "Done"
        print "==================================================================================\n"

    def transferTrainToLog(self):
        self.trainDF['SalePrice'] = np.log1p(self.trainDF['SalePrice'])


    #Features engineering

    def getJoinedDF(self):
        print "============================ get All DF=============================================="
        self.nTrain = self.trainDF.shape[0]
        self.nTest = self.testDF.shape[0]
        self.allDF = pd.concat((self.trainDF, self.testDF)).reset_index(drop=True)
        self.trainDF_label = self.trainDF['SalePrice']
        self.allDF.drop(['SalePrice'], axis=1, inplace=True)
        print "All DF size is", self.allDF.shape
        print "Done"
        print "===================================================================================\n"
    
    def printAllDF(self):
        print "============================ print allDF ============================================"
        print self.allDF
        print "Done"
        print "============================ print allDF ==========================================\n"

    def fillMissingData(self):
        print "============================ Fill missing data  ====================================="
        count = self.allDF.isnull().sum().sort_values(ascending=False)
        ratio = count/len(self.allDF)
        print "[Before]Top Na Columns:\n",ratio.head(20)
        print "Removing columns whose ratio>0.8 :", ratio[ratio > 0.8].index
        self.allDF = self.allDF.drop(ratio[ratio > 0.8].index, axis=1)
        print "Done removing"
        #self.allDF['PoolQC'] = self.allDF['PoolQC'].fillna('None')
        #self.allDF['MiscFeature'] = self.allDF['MiscFeature'].fillna('None')
        #self.allDF['Alley'] = self.allDF['Alley'].fillna('None')
        #self.allDF['Fence'] = self.allDF['Fence'].fillna('None')
        self.allDF['FireplaceQu'] = self.allDF['FireplaceQu'].fillna('None')
        self.allDF["LotFrontage"] = self.allDF.groupby("Neighborhood")["LotFrontage"].transform(
                lambda x: x.fillna(x.median()))
        for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
                self.allDF[col] = self.allDF[col].fillna('None')
        for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
                self.allDF[col] = self.allDF[col].fillna(0)
        for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
                self.allDF[col] = self.allDF[col].fillna(0)
        for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
                self.allDF[col] = self.allDF[col].fillna('None')
        self.allDF["MasVnrType"] = self.allDF["MasVnrType"].fillna("None")
        self.allDF["MasVnrArea"] = self.allDF["MasVnrArea"].fillna(0)
        self.allDF['MSZoning'] = self.allDF['MSZoning'].fillna(self.allDF['MSZoning'].mode()[0])
        self.allDF = self.allDF.drop(['Utilities'], axis=1)
        self.allDF["Functional"] = self.allDF["Functional"].fillna("Typ")
        self.allDF['Electrical'] = self.allDF['Electrical'].fillna(self.allDF['Electrical'].mode()[0])
        self.allDF['KitchenQual'] = self.allDF['KitchenQual'].fillna(self.allDF['KitchenQual'].mode()[0])
        self.allDF['Exterior1st'] = self.allDF['Exterior1st'].fillna(self.allDF['Exterior1st'].mode()[0])
        self.allDF['Exterior2nd'] = self.allDF['Exterior2nd'].fillna(self.allDF['Exterior2nd'].mode()[0])
        self.allDF['SaleType'] = self.allDF['SaleType'].fillna(self.allDF['SaleType'].mode()[0])
        self.allDF['MSSubClass'] = self.allDF['MSSubClass'].fillna("None")
        count = self.allDF.isnull().sum().sort_values(ascending=False)
        ratio = count/len(self.allDF)
        print "[After]Top Na Columns:\n",ratio.head(20)
        print "Done"
        print "====================================================================================\n"

        print "============================== drop columns  ========================================"
        print "Before drop: ",self.allDF.shape
        '''self.allDF = self.allDF.drop([ 'RoofMatl', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',\
                         'Heating', 'LowQualFinSF','BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt',\
                         'GarageArea', 'GarageCond', 'WoodDeckSF','OpenPorchSF', 'EnclosedPorch', '3SsnPorch',\
                         'ScreenPorch', 'PoolArea', 'MiscVal'],\
                         axis=1) '''
        print "After drop: ",self.allDF.shape
        print "====================================================================================\n"

    def transferNumToCate(self):
        print "===================== transfer number to category ===================================="
        self.allDF['MSSubClass'] = self.allDF['MSSubClass'].apply(str)
        self.allDF['OverallCond'] = self.allDF['OverallCond'].astype(str)
        self.allDF['YrSold'] = self.allDF['YrSold'].astype(str)
        self.allDF['MoSold'] = self.allDF['MoSold'].astype(str)
        print "Done"
        print "====================================================================================\n"

    def transferCateToLabelencoder(self): 
        print "===================== transfer category to labelEcoder ==============================="
        from sklearn.preprocessing import LabelEncoder
        cateCols = ['FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond',\
                'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', \
                'BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure',\
                'GarageFinish', 'LandSlope','LotShape', 'PavedDrive', 'Street', 'Alley',\
                'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold']
        for col in set(cateCols).intersection(self.allDF.columns):
            lbl = LabelEncoder()
            lbl.fit(list(self.allDF[col].values))
            self.allDF[col] = lbl.transform(list(self.allDF[col].values))
        print "Done"
        print "======================================================================================\n"


    def createMoreFeature(self):
        self.allDF['TotalSF'] = self.allDF['TotalBsmtSF'] + self.allDF['1stFlrSF'] + self.allDF['2ndFlrSF']


    def transformSkewedFeatures(self):
        print "========================= transform skewed data ======================================="
        numeric_feats = self.allDF.dtypes[self.allDF.dtypes != "object"].index
        skewed_feats = self.allDF[numeric_feats].apply(lambda x: x.dropna().skew()).sort_values(ascending=False)
        print("\nSkew in numerical features: \n")
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        print "before:", skewness.shape

        skewness = skewness[abs(skewness) > 0.75].dropna()
        print "Transforming whose absolute skewness > 0.75"

        print("{} skewed numerical features to be tranformed with Box Cox".format(skewness.shape[0]))

        from scipy.special import boxcox1p
        skewed_features = skewness.index
        lam = 0.15
        for feat in skewed_features:
                self.allDF[feat] = boxcox1p(self.allDF[feat], lam)
        print "Done"
        print "======================================================================================\n"

    def getDummyAllDF(self):
        self.dm_allDF = pd.get_dummies(self.allDF)
        
    def printDummytAllDF(self):
        print "===============================print Dummy AllDF======================================="
        print self.dm_allDF
        print "Done"
        print "=====================================================================================\n"

    def getDummyTrainTestDF(self):
        self.dm_trainDF = self.dm_allDF[:self.nTrain]
        self.dm_testDF = self.dm_allDF[self.nTrain:]


    def getTrainTestValues(self): 
        self.X_train = self.dm_trainDF.drop(['Id'],axis=1).values
        self.Y_train = self.trainDF_label.values
        
        self.X_test_ids = self.dm_testDF['Id'].values
        self.X_test = self.dm_testDF.drop(['Id'],axis=1).values


    def createModel(self):
        print "============================Create Train Model=========================================="
        self.xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05)
        print "Done"
        print "======================================================================================\n"

    def train(self):
        print "===================================Train================================================"
        self.xgb.fit(self.X_train,self.Y_train)#,early_stopping_rounds=5)
        print "Done"
        print "======================================================================================\n"

    def predict(self):
        print "===================================Predict=============================================="
        self.predict = np.expm1(self.xgb.predict(self.X_test))
        print "Done"
        print "======================================================================================\n"

    def savePrediction(self):    
        print "================================Save Prediction========================================="
        filename = "results.csv"
        for n in range(1,1000):
            filename = "results_{0}.csv".format(n)
            if not os.path.exists(filename):
                break
        with open(filename,'w') as f:
            f.write("Id,SalePrice\n")
            for i in range(len(self.predict)):
                f.write("{0},{1}\n".format(self.X_test_ids[i],self.predict[i]))
        print "Done"
        print "======================================================================================\n"


    
    
 
if __name__ == "__main__":
    predictor = housePricePredictor()
    predictor.getTrainTestDF()
    try: predictor.drawByTrainDF()
    except: pass
    predictor.dropRowsTrainDF()
    predictor.transferTrainToLog()
    predictor.getJoinedDF()
    predictor.fillMissingData()
    predictor.transferNumToCate()
    predictor.transferCateToLabelencoder()
    predictor.createMoreFeature()
    predictor.transformSkewedFeatures()
    predictor.getDummyAllDF()
    #predictor.printAllDF()
    #predictor.printDummytAllDF()
    predictor.getDummyTrainTestDF()
    predictor.getTrainTestValues()
    predictor.createModel()
    predictor.train()
    predictor.predict()
    predictor.savePrediction()

