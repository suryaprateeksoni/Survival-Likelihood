# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 18:54:00 2020

@author: Surya Prateek Soni
"""
################################################################################
# Working Directory
################################################################################
import os
os.getcwd()
os.chdir("C:\\Users\\Surya Prateek Soni\\Desktop")

################################################################################
# Importing Libraries
################################################################################
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


# Reading the Training and Testing dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# Combining both Train and Test for the operations where 
combine = [train_data, test_data]

# Previewing the data
train_data.head()
train_data.tail()
# Which features are available in the dataset?
# Getting all the columns 
print(train_data.columns.values)


# Which features are categorical? and which continous and which Mixed ?
# Which features may contain errors or typos?
# Which features contain blank, null or empty values?
# What are the data types for various features?
train_data.info()
print('_'*40)
test_data.info()


# What is the distribution of numerical feature values across the samples?
# Information about Statistical - Mean, Median, Variance, percentile for all columns 
train_data.describe()


################################################################################
# EDA 
################################################################################
# Correlation - how well does each feature correlate with Dependent Variable 
# Class VS Surbival Rate 
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Sex VS Surbival Rate 
train_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Siblings VS Surbival Rate 
train_data[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
# Parents VS Surbival Rate 
train_data[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)


################################################################################
# Analyze by visualizing data
################################################################################
# Histogram of Age VS Survived 
g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# Histogram of Age and Survived VS Class
grid = sns.FacetGrid(train_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

# Line Chart of Class and Embarked and Sex VS Survived
grid = sns.FacetGrid(train_data, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()

# Barplot of Sex and Embarked and Fare VS Survived 
grid = sns.FacetGrid(train_data, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()


################################################################################
# Wrangle data 
################################################################################
# Drop the Cabin and Ticket features because they are not useful for us.
# Getting the shapes of Train and Test 
print("Before", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape)
train_data = train_data.drop(['Ticket', 'Cabin'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_data, test_data]
"After", train_data.shape, test_data.shape, combine[0].shape, combine[1].shape

# Creating new feature extracting from existing
# Extract titles from names and test correlation between titles and survival
# Title Vs Sex
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)   # expand = False means it will return a DatraFrame 
pd.crosstab(train_data['Title'], train_data['Sex'])

# Replace many titles with a more common name "Rare"
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')    
train_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# Converting the categorical titles to ordinal - Mr/Miss/Mrs to 0,1,2
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
train_data.head()

# Droping the "Name" and "PassengerID" feature from training and testing datasets
# We are dropping only from train dataset, not from Test because we need that column for submitting our result. 
train_data = train_data.drop(['Name', 'PassengerId'], axis=1)
test_data = test_data.drop(['Name'], axis=1)
combine = [train_data, test_data]
train_data.shape, test_data.shape


# Converting a categorical feature - strings to numerical values
# Converting Sex into gender 0 and 1 
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
train_data.head()


# Completing a numerical continuous feature
# Age, Gender, and Pclass are correlated. We have to guess Age values using Median of sets of Gender and Pclass
grid = sns.FacetGrid(train_data, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

# Prepare an empty array to contain guessed Age values based on Pclass x Gender combinations.
guess_ages = np.zeros((2,3))
guess_ages
# Confirming that there are Null values in the feature "Age"
train_data.info()
print('_'*40)
test_data.info()
# Iterate over Sex (0 or 1) and Pclass (1, 2, 3) to calculate guessed values of Age for the six combinations.
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()
            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)
            age_guess = guess_df.median()
            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]
    dataset['Age'] = dataset['Age'].astype(int)
train_data.head()
# Confirming that there are NO Null values in the feature "Age"
train_data.info()
print('_'*40)
test_data.info()

# Create Age bands, so that we can convert them into ordinals later on, and determine correlations with Survived.
train_data['AgeBand'] = pd.cut(train_data['Age'], 5)
train_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# Replace Age with ordinals based on these bands
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_data.head()

# Remove the Age band feature 
train_data = train_data.drop(['AgeBand'], axis=1)
combine = [train_data, test_data]
train_data.head()



# Create new feature combining existing features
# Combine Parch and SibSp and then drop these individual features 
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
train_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Creating another feature called IsAlone 
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
train_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

# Now becasue we have got "IsAlone", we can drop Parch', 'SibSp', 'FamilySize
train_data = train_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_data = test_data.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_data, test_data]
train_data.head()

# Because Age and Class are individually correlated, lets us create an Artificial Feature with their combonation
# New Feature = Age * Class
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
train_data.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)



# Completeting a Categorical feature
# Embarked Feature has missing values, we have to fill them. Because there is no Correlation, hence, fill them with most frequent value (Mode)
freq_port = train_data.Embarked.dropna().mode()[0]
freq_port
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)    
train_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Converting categorical feature to numeric
# Converting Categorical Embarked into Ordinal Embarked. Replacing S,C,Q with 0,1,2
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
train_data.head()



# Completing and converting a numeric feature
# "Fare" Feature has 1 missing values, hence, fill them with most frequent value (Mode)
test_data['Fare'].fillna(test_data['Fare'].dropna().median(), inplace=True)
test_data.head()

# Now because Fare and Class are correlated. Class and Survival are Correlated. 
# So it might be possible that Fare and Survival is also Correlated 
# So, creating Fare bands, then convert them into Ordinals
train_data['FareBand'] = pd.qcut(train_data['Fare'], 4)
train_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
# It shows as Fare bands increase, survival rate also increases

# Convert the Fare feature to ordinal values based on the FareBand
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train_data = train_data.drop(['FareBand'], axis=1)
combine = [train_data, test_data] 
train_data.head(10)
test_data.head(10)



################################################################################
# Model Building 
################################################################################
# Model, predict and solve
# Spliting 
x_train= train_data.drop("Survived", axis=1)
y_train = train_data["Survived"]
x_test  = test_data.drop("PassengerId", axis=1).copy()
x_train.shape, y_train.shape, x_test.shape



# Logistic Regression
logreg = LogisticRegression()  # Creating Instance 
logreg.fit(x_train, y_train)
y_predicted= logreg.predict(x_test)
acc_log = round(logreg.score(x_train, y_train) * 100, 2)  # Finding the score 
acc_log
# Calculating the coefficient of the features 
# Positive coefficients increase the log-odds of the response (and thus increase the probability), and negative coefficients decrease the log-odds of the response (and thus decrease the probability).
coeff_data = pd.DataFrame(train_data.columns.delete(0))
coeff_data.columns = ['Feature']
coeff_data["Correlation"] = pd.Series(logreg.coef_[0])
coeff_data.sort_values(by='Correlation', ascending=False)

# Support Vector Machines
svc = SVC()
svc.fit(x_train, y_train)
y_predicted= svc.predict(x_test)
acc_svc = round(svc.score(x_train, y_train) * 100, 2)
acc_svc

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_predicted= knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
acc_knn

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_predicted= gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
acc_gaussian

# Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_predicted= perceptron.predict(x_test)
acc_perceptron = round(perceptron.score(x_train, y_train) * 100, 2)
acc_perceptron

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_predicted= linear_svc.predict(x_test)
acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)
acc_linear_svc


# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_predicted= sgd.predict(x_test)
acc_sgd = round(sgd.score(x_train, y_train) * 100, 2)
acc_sgd


# Decision Trees
decision_tree = DecisionTreeClassifier()
decision_tree.fit(x_train, y_train)
y_predicted= decision_tree.predict(x_test)
acc_decision_tree = round(decision_tree.score(x_train, y_train) * 100, 2)
acc_decision_tree


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_predicted= random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
acc_random_forest

################################################################################3
# Model Evaluation 
################################################################################3

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree]})
models.sort_values(by='Score', ascending=False)


submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": Y_pred
    })
# submission.to_csv('../output/submission.csv', index=False)