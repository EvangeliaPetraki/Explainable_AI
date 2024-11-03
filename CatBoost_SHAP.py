#importing necessary libraries for the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from catboost import MetricVisualizer
from catboost import cv
from imblearn.over_sampling import SMOTE
import shap

%matplotlib inline
shap.initjs()


#loading our dataset
path='C:\\Users\\rozat\\dataset.csv'
raw_data=pd.read_csv(path)

#we check the heads of the colunms and the context of the firtst 10 rows
raw_data=raw_data.reset_index(drop = True)
raw_data.head(10)

data = raw_data.copy()
data.isnull().sum() #check the dataset for missing values
data = data.astype("category")

#balancing our data
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state = 42)

target = data['UnderRisk']
features = data.drop(['UnderRisk'], axis =1)

features_sm, target_sm = sm.fit_resample(features, target)

features = pd.DataFrame(features_sm, columns=features.columns)
data_sm=pd.concat([features, target_sm], axis =1)

#setting which variables are fearures and which variable is the target.
UnderRisk = data_sm['UnderRisk']
features = data_sm.drop(['UnderRisk'], axis = 1)

#Categorical features declaration
cat_features = list(range(0, features.shape[1]))
print(cat_features)

#Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(features,UnderRisk, test_size = 0.2, random_state = 42)

#CatBoost implementation. First model with 5 iterations
cb_model_5 = CatBoostClassifier(iterations = 5, #the number of boosting iterations or trees that will be built during the training process.
                        learning_rate = 0.1)

cb_model_5.fit(X_train,y_train,
            cat_features = cat_features,
            eval_set = (X_test, y_test), #specify a validation set for monitoring the model's performance during training
            verbose = True) # Display progress and performance information during training

print('CatBoost model is fitted: ' + str(cb_model_5.is_fitted()))
print('CatBoost model parameters:')
print(cb_model_5.get_params())

#CatBoost implementation. Second model with 10 iterations
cb_model_10 = CatBoostClassifier(iterations = 10, #the number of boosting iterations or trees that will be built during the training process.
                        learning_rate = 0.1)

cb_model_10.fit(X_train,y_train,
            cat_features = cat_features,
            eval_set = (X_test, y_test), #specify a validation set for monitoring the model's performance during training
            verbose = True) # Display progress and performance information during training

print('CatBoost model is fitted: ' + str(cb_model_10.is_fitted()))
print('CatBoost model parameters:')
print(cb_model_10.get_params())

#CatBoost implementation. Third model with 15 iterations
cb_model_15 = CatBoostClassifier(iterations = 15, #the number of boosting iterations or trees that will be built during the training process.
                        learning_rate = 0.1)

cb_model_15.fit(X_train,y_train,
            cat_features = cat_features,
            eval_set = (X_test, y_test), #specify a validation set for monitoring the model's performance during training
            verbose = True) # Display progress and performance information during training

print('CatBoost model is fitted: ' + str(cb_model_15.is_fitted()))
print('CatBoost model parameters:')
print(cb_model_15.get_params())

# Model Predictions for the third model
print(cb_model_15.predict_proba(X_test))

# Metrics calculation and graph plotting 
cb_model_15 = CatBoostClassifier(iterations = 15,
                             random_seed =63 ,  #Sets the random seed for reproducibility
                             learning_rate = 0.1,
                             custom_loss = ['Accuracy','AUC']) # the custom loss is set to 'Accuracy', which indicates that the model will be trained to maximize accuracy

cb_model_15.fit(X_train,y_train,
            cat_features = cat_features,
            eval_set = (X_test,y_test),
            logging_level = 'Silent',
            plot = True)

# Import necessary libraries
from catboost import CatBoostClassifier,Pool
from catboost import Pool


# SHAP values calculation
explainer = shap.TreeExplainer(cb_model_15)
shap_values = explainer.shap_values(X_test)

# Visualize a single prediction (replace 0 with the index of the instance you want to visualize)
shap.initjs() #Initializing the SHAP JavaScript visualizations
shap.force_plot(
    explainer.expected_value,
    shap_values[13],
    X_test.iloc[13],
    matplotlib=True)

#The summary bar plot is a horizontal bar chart that shows the average absolute SHAP values for each feature, ordered by importance.
shap.summary_plot(shap_values, X_test, plot_type="bar")

explainer = shap.Explainer(cb_model_15)
shap_values = explainer(X_test)

np.shape(shap_values.values)

#waterfall plot for the first observation
shap.plots.waterfall(shap_values[0])

shap.plots.beeswarm(shap_values)
