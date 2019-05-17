import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
# https://www.kaggle.com/c/titanic


class titanic:
    def __init__(self):
        self.train_df = pd.read_csv("train.csv")
        self.train_label_data = self.train_df["Survived"].values
        self.train_data_df = self.train_df.drop(["Survived"],axis=1)
        self.test_df = pd.read_csv("test.csv")
        self.train_test_df = pd.concat([self.train_data_df,self.test_df],axis=0)
        self.test_df_ids = self.test_df['PassengerId'].values

    def featureEngineering(self):
        # Name
        def findTittle(name):
            title = name.split(", ")[1].split(".")[0]
            return title if title else None
        self.train_test_df["Name"] = self.train_test_df["Name"].apply(lambda x: findTittle(x) if findTittle(x) else x)

        # Age
        name_group = self.train_test_df.groupby("Name")
        age_by_name = name_group['Age']
        self.train_test_df["Age"] = age_by_name.transform(lambda x: x.fillna(x.median())) 
        self.train_test_df["Age"] =  pd.cut(self.train_test_df["Age"],[0,10,20,30,40,50,60,70,80,90])

        # Ticket 
        ticket_group_df = self.train_test_df.groupby("Ticket",as_index = False)['PassengerId'].count()
        one_ticket_df = ticket_group_df[ticket_group_df['PassengerId']==1]
        one_ticket_series = one_ticket_df['Ticket']
        self.train_test_df['Ticket'] = np.where(self.train_test_df['Ticket'].isin(one_ticket_series),0, 1)
        # Fare
        self.train_test_df['Fare'] = self.train_test_df['Fare'].fillna(0)
        self.train_test_df['Fare'] = pd.cut(self.train_test_df['Fare'],[-1,50,100,150,200,250,300,350,400,450,500,550])

        # Cabin
        self.train_test_df['Cabin'] = self.train_test_df['Cabin'].fillna("No")
        self.train_test_df['Cabin'] = np.where(self.train_test_df['Cabin']=='No',0,1)

        # Embarked
        self.train_test_df['Embarked'] = self.train_test_df['Embarked'].fillna(self.train_test_df['Embarked'].mode()[0])

        # Create Family
        self.train_test_df['Family'] = self.train_test_df['SibSp'] + self.train_test_df['Parch'] + 1
    
        # drop features
        self.train_test_df = self.train_test_df.drop(["PassengerId","SibSp","Parch"],axis=1)
        
        # get numberic values
        self.train_test_df['Name'] = self.train_test_df['Name'].map({name:i for i,name in enumerate(self.train_test_df["Name"].unique())})

        self.train_test_df['Sex'] = self.train_test_df['Sex'].map({name:i for i,name in enumerate(self.train_test_df["Sex"].unique())})

        self.train_test_df['Age'] = self.train_test_df['Age'].map({name:i for i,name in enumerate(self.train_test_df["Age"].unique())})

        self.train_test_df['Fare'] = self.train_test_df['Fare'].map({name:i for i,name in enumerate(self.train_test_df["Fare"].unique())})

        self.train_test_df['Embarked'] = self.train_test_df['Embarked'].map({name:i for i,name in enumerate(self.train_test_df["Embarked"].unique())})

    def splitTrainTest(self):
        self.train_data_df = self.train_test_df[:len(self.train_data_df)]
        self.test_df = self.train_test_df[len(self.train_data_df):]

    def train(self):
        X_train, X_valid, Y_train, Y_valid = train_test_split(self.train_data_df.values, self.train_label_data, test_size=0.2)
        self.rft = RandomForestClassifier(random_state=30)
        self.rft.fit(X_train,Y_train)
        print self.rft.score(X_valid,Y_valid)
        self.predict = self.rft.predict(self.test_df.values)

    def savePrediction(self,filename):
        with open(filename,'w') as f:
            f.write("PassengerId,Survived\n")
            for i in range(len(self.predict)):
                f.write("{0},{1}\n".format(self.test_df_ids[i],self.predict[i]))

if __name__ == "__main__":
    obj = titanic()
    obj.featureEngineering()
    obj.splitTrainTest()
    obj.train()
    obj.savePrediction("result.csv")
