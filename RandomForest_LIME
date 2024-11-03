import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

#in order for the plots to be produced in the notebooks and not in another window
%matplotlib inline

#in order to load the dataset from drive
from google.colab import drive
drive.mount('/content/drive')

np.random.seed(12345)

#loading the dataset

path = '/content/drive/MyDrive/Colab Notebooks/Dataset.csv'
raw_data = pd.read_csv(path, na_values='?')

raw_data=raw_data.reset_index(drop = True)
raw_data.head(10)

#Looking for Null values

data = raw_data.copy()
data.isnull().sum()

#Checking whether the dataset is balanced

samples_yes=data[data['UnderRisk']=='yes']
samples_no =data[data['UnderRisk']=='no']

all_samps = {'Under Risk' : [samples_yes.shape[0]], 'Not Under Risk' : [samples_no.shape[0]]}

both_df=pd.DataFrame(data= all_samps)
both_df.plot(kind='bar', figsize = (10,4))

#not balanced -> SMOTE to balance them

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 42)

target = data['UnderRisk']
features = data.drop(['UnderRisk'], axis =1)

features_sm, target_sm = sm.fit_resample(features, target)

features = pd.DataFrame(features_sm, columns=features.columns)
data_sm=pd.concat([features, target_sm], axis =1)

#Checking whether they are balanced now

samples_yes=data_sm[data_sm['UnderRisk']=='yes']
samples_no =data_sm[data_sm['UnderRisk']=='no']

all_samps = {'Under Risk' : [samples_yes.shape[0]], 'Not Under Risk' : [samples_no.shape[0]]}

both_df=pd.DataFrame(data= all_samps)
both_df.plot(kind='bar', figsize = (10,4))

#Exploratory data analysis

samples_male=data[data['Gender']==1]
samples_female =data[data['Gender']==2]
male_and_fem = {'Males' : [samples_male.shape[0]], 'Females' : [samples_female.shape[0]] }
gender_df=pd.DataFrame(data= male_and_fem)
gender_df.plot(kind='bar', figsize = (10,4))

data['Obese'].value_counts()

samples_obese=data[data['Obese']==1]
samples_not_obese =data[data['Obese']==0]

Obese = {'Obese' : [samples_obese.shape[0]], 'Not Obese' : [samples_not_obese.shape[0]] }

obese_df=pd.DataFrame(data= Obese)
obese_df.plot(kind='bar', figsize = (10,4))

data['HighBP'].value_counts()

samples_HighBP=data[data['HighBP']==1]
samples_NormalBP =data[data['HighBP']==0]

BP = {'High BP' : [samples_HighBP.shape[0]], 'Normal BP' : [samples_NormalBP.shape[0]] }

BP_df=pd.DataFrame(data= BP)
BP_df.plot(kind='bar', figsize = (10,4))

data['Use_of_stimulant_drugs'].value_counts()
data['History_of_preeclampsia'].value_counts()
data['Metabolic_syndrome'].value_counts()
data['Diabetes'].value_counts()
data['Diabetes'].value_counts()
data['Chain_smoker'].value_counts()
data['Respiratory_illness'].value_counts()

#Seeing the structure of the data
data_sm.describe()

#saving the target and the Feature values to variables
UnderRisk = data_sm['UnderRisk']
features = data_sm.drop(['UnderRisk'], axis = 1)

#splitting the model to train and test sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, UnderRisk, test_size = 0.2, random_state = 42)

#droping the indexes
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

#saving the values separately
X_train_values = X_train.values
X_test_values = X_test.values
y_train_values = y_train.values
y_test_values = y_test.values

X_train_values.shape, X_test_values.shape, y_train_values.shape, y_test_values.shape,

#importing random forest
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

rfc5 = RandomForestClassifier(n_estimators=5)
rfc10 = RandomForestClassifier(n_estimators=10)
rfc50 = RandomForestClassifier(n_estimators=50)

#training the models

#model with 5 trees
X_train_processed = X_train_values
X_test_processed = X_test_values
print(X_train_processed.shape, y_train_values.shape, X_test_processed.shape, y_test_values.shape)
rfc5.fit(X_train_processed, y_train)
score5=rfc5.score(X_test_processed, y_test)
print("score5 = ", score5)

#model with 10 trees
X_train_processed = X_train_values
X_test_processed = X_test_values
print(X_train_processed.shape, y_train_values.shape, X_test_processed.shape, y_test_values.shape)
rfc10.fit(X_train_processed, y_train)
score10=rfc10.score(X_test_processed, y_test)
print("score10 = ", score10)

#model with 50 trees
X_train_processed = X_train_values
X_test_processed = X_test_values
print(X_train_processed.shape, y_train_values.shape, X_test_processed.shape, y_test_values.shape)
rfc50.fit(X_train_processed, y_train)
score50=rfc50.score(X_test_processed, y_test)
print("score50 = ", score50)

#EXPLAINABILITY

#Feature importance map
feature_importances_map = {'importance' : rfc50.feature_importances_}
feature_importances_df = pd.DataFrame(feature_importances_map)
feature_importances_df.index = X_train.columns
feature_importances_df.sort_values('importance', ascending = True, inplace = True)
feature_importances_df

feature_importances_df.plot.barh(rot=0)

#LIME
from lime import lime_text
from sklearn.pipeline import make_pipeline
from lime.lime_tabular import LimeTabularExplainer
machine_learning_pipeline = make_pipeline(rfc50)

class_names = ['yes', 'no']
feature_names = ['Gender',	'Chain_smoker',	'Consumes_other_tobacco_products',	'HighBP',	'Obese',	'Diabetes',	'Metabolic_syndrome',	'Use_of_stimulant_drugs',	'Family_history',	'History_of_preeclampsia',	'CABG_history',	'Respiratory_illness']

explainer = LimeTabularExplainer(feature_names = feature_names,
                                 class_names = class_names,
                                 training_data = X_train_values)

#Plot function
def plot(exp, label = 1):
  exp_list = exp.as_list()
  fig = plt.figure()
  vals = [x[1] for x in exp_list]

  names = [x[0] for x in exp_list]

  vals.reverse()
  names.reverse()
  colors = ['green' if x <=0 else 'red' for x in vals]
  pos = np.arange(len(exp_list)) + .5
  plt.barh(pos, vals, align='center', color = colors)
  plt.yticks(pos, names)
  if exp.mode=='classification':
    title='Local explanation for class {}'.format(exp.class_names[label])
  else:
    title = 'Local explanation'
  plt.title = title
  return fig

#first sample for prediction
sample_1 = raw_data.loc[5, feature_names]
type(sample_1), sample_1

p = rfc50.predict_proba(sample_1.values.reshape(1, -1))
p

print('\nfeature_vector:\n')
sample_1

#plotting the features that influenced the decision
p = machine_learning_pipeline.predict_proba(sample_1.values.reshape(1, -1))
exp = explainer.explain_instance(sample_1, rfc50.predict_proba, num_features = 12)

print('\nProbability of not Having a Risk', machine_learning_pipeline.predict_proba(sample_1.values.reshape(1, -1))[0,1])
print('True Class : {}'.format(class_names[int(round(p[0][1]))]))

fig = plot(exp,1)
exp.show_in_notebook(show_table=True, show_all=False)

#secind sample for prediction
sample_2 = raw_data.loc[10, feature_names]

#second prediction and plot
p = machine_learning_pipeline.predict_proba(sample_2.values.reshape(1, -1))
exp = explainer.explain_instance(sample_2, rfc50.predict_proba, num_features = 12)

print('\nProbability of not Having a Risk', machine_learning_pipeline.predict_proba(sample_2.values.reshape(1, -1))[0,1])
print('True Class : {}'.format(class_names[int(round(p[0][1]))]))

fig = plot(exp,1)
exp.show_in_notebook(show_table=True, show_all=False)
