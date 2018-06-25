import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
# https://www.kaggle.com/c/titanic


class titanic:
    def __init__(self):
        self.train_csv = "train.csv"
        self.test_csv = "test.csv"

    def getTrainTestData(self):
        self.train_df = pd.read_csv(self.train_csv)
        self.test_df = pd.read_csv(self.test_csv)
        self.train_df_data = self.train_df.drop(["Survived"],axis=1)
        self.train_df_label = self.train_df[["Survived"]]
    
    def getDummy(self):
        self.train_df_data_dummy = pd.get_dummies(self.train_df_data) 
        self.test_df_dummy  = pd.get_dummies(self.test_df)

    def setTestColumnsAsTrain(self):
        trainCols = self.train_df_data_dummy.columns.values.tolist()
        self.test_df_dummy = self.test_df_dummy.loc[:,trainCols]

    def resolveNa(self):
        self.train_df_data_dummy = self.train_df_data_dummy.fillna(0)
        self.test_df_dummy = self.test_df_dummy.fillna(0)

    def getValues(self):
        self.train_df_data_dummy_values = self.train_df_data_dummy.values
        self.test_df_dummy_values = self.test_df_dummy.values
        self.train_df_label_values = self.train_df_label.values
        self.test_df_ids = self.test_df['PassengerId'].values

    def train(self):
        self.rft = RandomForestClassifier(random_state=1)
        self.rft.fit(self.train_df_data_dummy_values,self.train_df_label_values.ravel())
        self.predict = self.rft.predict(self.test_df_dummy_values)

    def savePrediction(self,filename):
        with open(filename,'w') as f:
            f.write("PassengerId,Survived\n")
            for i in range(len(self.predict)):
                f.write("{0},{1}\n".format(self.test_df_ids[i],self.predict[i]))

if __name__ == "__main__":
    obj = titanic()
    obj.getTrainTestData()
    obj.getDummy()
    obj.setTestColumnsAsTrain()
    obj.resolveNa()
    obj.getValues()
    obj.train()
    obj.savePrediction("titanic_submission.csv")
    
