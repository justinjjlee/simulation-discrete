# -*- coding: utf-8 -*-
"""Exercise - Predicting the Churn.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fKwMyIlIzD3LJn0SiHbSvCqv1TtGusvU

# Predicting the churn - An example customer data

Using an example telecomm subscription data in building models predicting customer churns, the following exercise showcase,

* Importance of addressing data imbalance, such as customer churn data, where data consists of significantly smaller subset of observations with success features
  * Implication on decisions and an example of making a 'judgement call' based on first-order objectives
* Leveraging Shapley values to explain and prescribe business recommendations
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils import resample

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
# For visualizing the tree
from sklearn.tree import export_graphviz
from sklearn import tree

!pip install shap -q
import shap

# Import data from the public domain
df = pd.read_csv("https://raw.githubusercontent.com/erood/interviewqs.com_code_snippets/master/Datasets/teleco_user_data.csv")
df.head(3)

print(df.columns)

"""## Understanding the data and feature engineering

### Identifying variables and grouping the types
"""

# Binary choice with conditional default values (N/A)
print(df.MultipleLines.unique())
# Nested with column PhoneService

# Below are nested with information,
#   if the customer has the internet subscription or not.
print(df.OnlineSecurity.unique())
print(df.OnlineBackup.unique())
print(df.DeviceProtection.unique())
print(df.TechSupport.unique())
print(df.StreamingTV.unique())
print(df.StreamingMovies.unique())

print(df.InternetService.unique())
# If they have internet subscription, then there are two types and options.

# Woult first break down into internet service or not.
# then dissect with time
print(df.Contract.unique())
# Subscription can be based on a contract (one-year or two-year) or not

print(df.PaymentMethod.unique())
# Can be automatic or not
# Can be snail or electronic

"""Continuous variable, whcih can be translated to categorical variable or left as continuous variable."""

print(df.MonthlyCharges.unique()) # monthly charge (average)
print(df.TotalCharges.unique()) # lifetime value
print(df.tenure.unique()) # ordinal variable

"""#### Binary variables"""

print(df.gender.unique())
print(df.SeniorCitizen.unique()) # No need to translate, already binary
print(df.Partner.unique())
print(df.Dependents.unique())
print(df.PhoneService.unique())

# What the KPI/response variable would be.
print(df.Churn.unique())

dff = df.copy(deep = True)
# dataframe used for the analysis and model feed.

dff['gender'] = np.where(dff['gender'].str.contains('Female'), 1, 0)

# This is a nested column condition on 'PhoneService' column
dff['MultipleLines'] = np.where(dff['MultipleLines'].str.contains('Yes'), 1, 0)


# ..............................................................................
# Subscription: For those in contract, either monthly, one year, or two years
dff['Contract'].replace({'Month-to-month':'contract_mo', 
                         'One year':'contract_1yr',
                         'Two year':'contract_2yr'}, inplace = True)
# Use sklearn's encoder - create binary columns for 
lb = LabelBinarizer()
lb_results = lb.fit_transform(dff['Contract'])
lb_results_df = pd.DataFrame(lb_results, columns=lb.classes_)

dff = pd.concat([dff, lb_results_df], axis = 1)
# Drop the string column
dff.drop(columns = ['Contract'], inplace = True)

# Note that for models requiring orthogonality considerations, one should
#   consider removing 'contract_mo' column. The variation will be represented by
#   value zero's in both 'contract-1yr' and 'contract-2yr.'

# ..............................................................................
# Payment method:
# I am identifying two attributes relevant: electronic or automatic payment
# (a) Electronic or not
dff['pymt_elec'] = np.where(dff['PaymentMethod'].str.contains('Mailed check'), 0, 1)
# (b) automatic payment or not
dff['pymt_auto'] = np.where(dff['PaymentMethod'].str.contains('automatic'), 1, 0)
# Other data view may not be useful in interpreting the results of the churn trend
dff.drop(columns = ['PaymentMethod'], inplace = True)
# ..............................................................................
# For other already-binary columns:

# Generate column for internet service 
dff['InternetService_io'] = np.where(dff['InternetService'].str.contains('No'), 0, 1)
# What type of internet service, Fiber optics? (this column is nested to prev.)
dff['InternetService'] = np.where(dff['InternetService'].str.contains('Fiber optic'), 1, 0)

# For columns requiring binary encoding with Yes/No
list_binaryCols = ['Partner', 'Dependents', 'PhoneService',
                   'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                   'TechSupport', 'StreamingTV', 'StreamingMovies',
                   'PaperlessBilling'
                   ]
for iter in list_binaryCols:
    dff[iter] = np.where(dff[iter].str.contains('Yes'), 1, 0)
dff.head(5)

"""#### Continuous variables

I clean up the charges and tenure data and exploring variations to identify columns and informations to be included in the model,
"""

# data cleaning is needed for 'TotalCharges'
# Data cleaning in this case would be the hard way, with string value of '_' in
# the empty elements
def str_to_flt(iter):
    #Check if it is convertible, if not spit out what he problem is 
    try:
      #try to convert
      return float(iter)
    except: # in case of no value - default to zero
      return 0 

dff["TotalCharges"] = [str_to_flt(iter) for iter in dff.TotalCharges]

fig, axs = plt.subplots(1, 3, sharey = True, tight_layout = True)
n_bins = 10

# We can set the number of bins with the *bins* keyword argument.
axs[0].hist(dff.MonthlyCharges, bins=n_bins);
axs[0].title.set_text('Monthly Charge');
axs[1].hist(dff.TotalCharges, bins=n_bins);
axs[1].title.set_text('Lifetime Value');
axs[2].hist(dff.tenure, bins = n_bins);
axs[2].title.set_text('Tenure');

fig.show()

"""The colinear relationship between the lifetime value (total charges) and tenure (i.e. longer tenured customers have paid more for the services received) should be examined before considered as a variable of interest. 

I check to see if the lifetime value provides any new information compare to the monthly charge and tenure. The two measures can ladder up to the lifetime value if the charges were consistent over time. I calculate lifetime value per given tenure and evaluate if the correlation between 'MonthlyCharges' is high - which it is. I can replace the lifetime value measure for the given dataset with the two measure orthogonal and provides information needed.

From a qualitative point of view, I want to see if the lifetime value of a customer relates to the liklihood to churn. Because the data does not provide records of transactions over time, I limit the scope of the analysis to just point-in-time estimate. With the availability, panel/time-varying value of customers can be considered.
"""

dff['TotalCharges_avg'] = dff.TotalCharges / dff.tenure; # average value given the length of duration
dff[['MonthlyCharges', 'TotalCharges', 'tenure', 'TotalCharges_avg']].corr()

fig, axes = plt.subplots(ncols=3, figsize=(12, 5), sharey=True);
str_var_eval = ['MonthlyCharges', 'TotalCharges_avg','tenure','Churn'];
dff[str_var_eval].boxplot(by='Churn', return_type='axes', ax=axes);

fig.show()

dff.drop(columns = ['TotalCharges', 'TotalCharges_avg'], inplace = True)

print(dff['MonthlyCharges'].describe())
print(dff['tenure'].describe())

"""Resources:
[DataCamp - Categorical data transformation](https://www.datacamp.com/community/tutorials/categorical-data)

### Defining the KPI
I use 'Churn' column - a binary measure - to be the measurement of interest (response variable).
"""

dff['Churn'] = np.where(dff['Churn'].str.contains('No'), 0, 1)
dff.groupby(['Churn']).Churn.count()

"""The data is not balanced - that there are significantly less number of individuals who churn than those who don't. For any pattern or trend models to work efficiently, we would need a reasonablly representative number of both outcomes for the model approach to properly learn existing patterns. 

Using an imbalanced data leads to models based on lack of representatitiveness of the behaviors or over-represent and generalize patterns that are overwhelmingly more common. For example, measure of accuracy (i.e. an overall measure of fitness of captured trends) may be high. But, once falsely identified predictions (falls positive/negative) are accounted, results and insights from the models are not representative - which can be captured by measures such as precision and recall.

The most logical way to deal with is to go back to the (business) objectives related to the exercise. Below, I explain my choice of the measurement and evaluation strategy.

The final dataset to be used in the model,
"""

# Finally, drop the customer ID for the model evaluations
# (or can be used as row index)
dff.drop(columns = 'customerID', inplace = True)
dff.head(5)

dff.corr().Churn.sort_values()

"""## Model construction and evaluation

### Measurement and evaluation strategy

From the application point of view, I prioritize the overall accuracy and recall over precision as my measurement and evaluation strategy for the predictive models evaluated below.

The primary objective of businesses is to prevent customers from churning (i.e. retention). The efforts required for businesses may need substantial investments - which identifying customers with high probability to churn with success would prioritize and optimize the investment. If there is model uncertainty, businesses need to prioritize and chose if they want to be conservative (not risk losing any customer) or risk-taking (they might be able to lose some customers, but they would not need to invest excessive budget on the retention efforts).

I assume that not able to detecte the churns may be costlier to the business than business cost of investing (e.g. discounts) on customers who would not churn. For the exercise below, I assume that the priority of the business is to prevent customers from churning, regardless of the cost associated with the customer retention efforts. This implies that we want to make sure that the measure of ***recall*** is high, as well as the accuracy.

For more complex approach, a second layer of predictive model can be used - instead of classification, build around probability of churn. Using the continuous predictor, one can prioritize customers to be engaged in the retention efforts.

NOTE: Documentation on [different model evaluation strategy](https://vitalflux.com/accuracy-precision-recall-f1-score-python-example/#What_is_F1-Score)

### Benchmark - logistic regression model

I use logistics regression as a baseline model framework.

#### (1) Baseline - how imbalanced data works
Following logistic regression model is defined through variable selection process. Note that colinearity should be concerned when fitting the model. For the sake of brevity, I do not go in depth of feature engineering, model identification, or variable selection here given its purpose of a simple benchmkaring process and providing an example of consequences of unattended imbalanced data. Following the process, I use the selected variables for the model construction,
"""

y = dff.Churn
x = dff.drop(columns=['Churn'])

# Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, 
                                                    random_state = 22)

x_str = ['tenure', 'PhoneService',  'MonthlyCharges', 'contract_mo', 
         'pymt_auto', 'InternetService']
x_trainx = x_train.loc[:, x_str]
x_testx = x_test.loc[:, x_str]

# Fit the model
clf = LogisticRegression(random_state=0).fit(x_trainx, y_train)

yhat_train = clf.predict(x_trainx)
yhat_test = clf.predict(x_testx)

# In case of continuous predictive probability,
#clf.predict_proba(x_trainx)

print(
    f'Training accuracy: {accuracy_score(y_train, yhat_train)}\n',
    f'Test accuracy: {accuracy_score(y_test, yhat_test)}\n\n',

    f'Training precision: {precision_score(y_train, yhat_train)}\n',
    f'Test precision: {precision_score(y_test, yhat_test)}\n\n',

    f'Training recall: {recall_score(y_train, yhat_train)}\n',
    f'Test recall: {recall_score(y_test, yhat_test)}\n'
)

"""Overall model fit,"""

# Select all data
y = dff.Churn
x = dff.drop(columns=['Churn'])

yhat_all = clf.predict(x[x_str])
print(
    f'Accuracy: {accuracy_score(y, yhat_all)}\n',
    f'Precision: {precision_score(y, yhat_all)}\n',
    f'Recall: {recall_score(y, yhat_all)}\n'
)

"""I show that while the accuracy of the model is high, the measured level of precision and recall is quite low. As mentioned above, based on the defined business objective, we want the measure of recall to be high. 

#### (2) Resampling to address imbalance of data

To build a better benchmark model, I force-balance the data so that the model may properly capture the variations in likelihood to churn. I use re-sampling method.

NOTE: Adjusting for balance, see [an example](https://elitedatascience.com/imbalanced-classes).
"""

from sklearn.utils import resample

dff_1 = dff.loc[dff.Churn == 1, :]
dff_0 = dff.loc[dff.Churn == 0, :] # over-represented

# For simplicty, sample equal length
# Upsample minority class
dff_0_downsampled = resample(dff_0, replace = False, n_samples = dff_1.shape[0], 
                             random_state = 22)
 
# Combine majority class with upsampled minority class
df_resampled = pd.concat([dff_1, dff_0_downsampled])

# Define response/explanatory variable
y = df_resampled.Churn
x = df_resampled.drop(columns=['Churn'])

# Divide up sample,
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, 
                                                    random_state = 22)

x_str = ['tenure', 'PhoneService',  'MonthlyCharges', 'contract_mo', 
         'pymt_auto', 'InternetService']
x_trainx = x_train.loc[:, x_str]
x_testx = x_test.loc[:, x_str]

# Fit the model
clf = LogisticRegression(random_state=0).fit(x_trainx, y_train)

yhat_train = clf.predict(x_trainx)
yhat_test = clf.predict(x_testx)

# Now, check to see if the data is balanced.
y_test.value_counts()

print(
    f'Training accuracy: {accuracy_score(y_train, yhat_train)}\n',
    f'Test accuracy: {accuracy_score(y_test, yhat_test)}\n\n',

    f'Training precision: {precision_score(y_train, yhat_train)}\n',
    f'Test precision: {precision_score(y_test, yhat_test)}\n\n',

    f'Training recall: {recall_score(y_train, yhat_train)}\n',
    f'Test recall: {recall_score(y_test, yhat_test)}\n'
)

"""Evaluate the overall fitness"""

# Select all data
y = dff.Churn
x = dff.drop(columns=['Churn'])

yhat_all = clf.predict(x[x_str])
print(
    f'Accuracy: {accuracy_score(y, yhat_all)}\n',
    f'Precision: {precision_score(y, yhat_all)}\n',
    f'Recall: {recall_score(y, yhat_all)}\n'
)

"""While the overall accuracy is slightly worsened, I now have significantly better recall measure. Following the defined business objective, the method addressing the imbalanced data is a required step in building the churn model."""

conf_matrix = confusion_matrix(y_true=y, y_pred=yhat_all)
#
# Print the confusion matrix using Matplotlib
# NOTE: the following code of printing the confusion matrix is copied from the
#   documentation of model fit measurements noted above.
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha = 0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', 
                size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

"""In layman's term, a lot of statistical or machine learning models default to what tips over (majority) when given features/information we explain with. This implies the use of the majority decision, tipping (logistic regression or activation functions for neural network, for example) or common trends (regression). In some cases, these imbalanced data/class can be treated as uncommon events - thus, we can approach the methods of anomaly detection.

### Another example: Decision Tree

I use decision tree with the re-sample and imbalance adjustment used above. Only parameter for a standard model that needs to be defined is the depth of the tree, which I start at 5. I repeat the simplified model building process shown above - but, in this case, I include all variables given the decision tree approach is resistent to the bias risks from the collinearity-related overfitting. One can repeat the exercise with more thought-out identification and feature selections - it does not impact overall fitness of the model.
"""

tree_clf = DecisionTreeClassifier(max_depth = 5)
tree_clf.fit(x_train, y_train)

# In case one wants to visualize the tree
txt_tree = tree.export_text(tree_clf)
#print(txt_tree) # uncomment to see

yhat_train = tree_clf.predict(x_train)
yhat_test = tree_clf.predict(x_test)

print(
    f'Training accuracy: {accuracy_score(y_train, yhat_train)}\n',
    f'Test accuracy: {accuracy_score(y_test, yhat_test)}\n\n',

    f'Training precision: {precision_score(y_train, yhat_train)}\n',
    f'Test precision: {precision_score(y_test, yhat_test)}\n\n',

    f'Training recall: {recall_score(y_train, yhat_train)}\n',
    f'Test recall: {recall_score(y_test, yhat_test)}\n'
)

"""The above shows that the imbalance-adjust data using the decision tree has the higher recall than the baseline model. Comparing the measurements with the benchmark logistic regression approach, the decision tree can be a better solution than the logistic regression. Observing the overall fitness,"""

# Select all data
y = dff.Churn
x = dff.drop(columns=['Churn'])

yhat_all = tree_clf.predict(x)
print(
    f'Accuracy: {accuracy_score(y, yhat_all)}\n',
    f'Precision: {precision_score(y, yhat_all)}\n',
    f'Recall: {recall_score(y, yhat_all)}\n'
)

conf_matrix = confusion_matrix(y_true=y, y_pred=yhat_all)
#
# Print the confusion matrix using Matplotlib
# NOTE: the following code of printing the confusion matrix is copied from the
#   documentation of model fit measurements noted above.
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha = 0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', 
                size='xx-large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()

"""## Reasoning and prescribing the churns

From the model constructed, some level of explanation and insights needs to be generated from the predicted churns. That includes interpreting the coefficients for the regression or reviewing the leafs for the tree method. For more coherent approach, I leverage Shapley values in explaining the model.
"""

explainer = shap.TreeExplainer(tree_clf)
shap_values = explainer.shap_values(x_train)

# visualize the prediction average scale of contribution
shap.summary_plot(shap_values, x_train)

"""To provide more directional interpretation, I dissect the contribution with directional feature values."""

#shap.summary_plot(shap_values[1], x_train)
shap.summary_plot(shap_values[1], x_train) # explaining the churn
# negative - contributing to 'no churn'
# positive - contributing to 'churn'

"""Based on the values, I can generalize that,

1.   Customers with short-term contracts are more at risk of churning (obviously)
  * Especially those under the monthly plan
  * This may also include customers with longer-term contracts that are about to expire
2.   Customers with add-on internet services (packages) tend to stay as retained customers
  * However, while a weak relationship, I observe higher likelihood to churn with those who spends more on the add-on features (streaming TV or Movies)

#### Recommended actions

The customer retention efforts should focus on customers who's contract expires within the 30-day window (monthly contract or longer term contracts that are about to expire with the period).

While high value customers tend to leverage the add-on features, excessive add-on's can lead to price burden for customers. Periodic price discount or offerings of the add-on features can help avoid churn risk of the high-valued customers.
"""