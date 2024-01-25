# Machine Learning for Stage Classification 
Author: Huang,Shu-Jing
Date: 2023-11-28

## environment
```shell
conda activate MachineLearning
cd /home/emily2835/EarlyLateStageProject/StagecClassification_MachineLearning
```

## Import packages
```python
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xg
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import compute_sample_weight
```
# [Defult model] performance for 5 models
## [Ligthgbm/XGBoost/RandomForest/logistic regression/SVC] 5 models and earlylate proggene to classify early and late stage for KIRC
```python
# Read data
KIRC_ELgene = pd.read_csv('Data/KIRCearlylatelabelCoxProggene005.csv', index_col=0)
KIRC_Eexpr = pd.read_csv('Data/KIRCearlystageExprandClin.csv', index_col=0)
KIRC_Lexpr = pd.read_csv('Data/KIRClatestageExprandClin.csv', index_col=0)
GENEcompare = pd.read_csv('Data/probeMap_gencode.v23.annotation.gene.probemap', sep='\t')
# Using regular expression to extract gene ensemble id with out version .1.2
GENEcompare['id'] = GENEcompare['id'].str.split('.').str[0]
# add label for early and late stage
KIRC_Eexpr['E_L_Stage'] = 'early'
KIRC_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRC_Expr = pd.concat([KIRC_Eexpr,KIRC_Lexpr],axis=0)
# check the number of early and late stage
KIRC_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage 
KIRC_GeneExpr = KIRC_Expr[KIRC_ELgene[KIRC_ELgene['ProgStage']=='earlylate']['gene']]
# Change colname(ensemble id) to gene name
KIRC_GeneExpr.columns = GENEcompare[GENEcompare['id'].isin(KIRC_GeneExpr.columns)]['gene'].values
KIRC_Target = KIRC_Expr['E_L_Stage']
# Log2 transformation
KIRC_GeneExpr = np.log2(KIRC_GeneExpr+1) # !!!!!
# Standardization 
scaler = StandardScaler() # !!!!!
KIRC_GeneExpr = scaler.fit_transform(KIRC_GeneExpr)
# Split data into training and testing sets
y = KIRC_Target
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(KIRC_GeneExpr,y,test_size=0.3, stratify=y,random_state=44) # !!!!!
w = compute_sample_weight(class_weight='balanced', y=y_train) 
# Defult parameters for 3 models
gbm = lgb.LGBMClassifier(random_state=55,boosting_type = 'gbdt',objective = 'binary', n_jobs = 50, metric = 'binary_logloss') # !!!!!
xg = xgb.XGBClassifier(use_label_encoder=True, eval_metric='logloss',random_state=55) # !!!!!
rf = RandomForestClassifier(random_state=55) # !!!!!
svc = SVC(kernel='linear', random_state=55)
lr = LogisticRegression(max_iter=200,random_state=55)
# 5-Fold Cross-validation scores
skf = StratifiedKFold(n_splits=10, random_state=44, shuffle=True)
scoresgbm = cross_val_score(gbm, X_train, y_train, cv=skf, scoring='accuracy')
scoresxg = cross_val_score(xg, X_train, y_train, cv=skf, scoring='accuracy')
scoresrf = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy')
scoresvc = cross_val_score(svc, X_train, y_train, cv=skf, scoring='accuracy')
scoreslr = cross_val_score(lr, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scoresgbm),'\n',
      'Cross-validation scores:{}'.format(scoresxg),'\n',
      'Cross-validation scores:{}'.format(scoresrf),'\n',
      'Cross-validation scores:{}'.format(scoresvc),'\n',
      'Cross-validation scores:{}'.format(scoreslr))
# Plotting the boxplot
df1 = pd.DataFrame({'Model': ['Lightgbm']*len(scoresgbm), 'Accuracy': scoresgbm})
df2 = pd.DataFrame({'Model': ['XGBoost']*len(scoresxg), 'Accuracy': scoresxg})
df3 = pd.DataFrame({'Model': ['Random Forest']*len(scoresrf), 'Accuracy': scoresrf})
df4 = pd.DataFrame({'Model': ['SVC']*len(scoresvc), 'Accuracy': scoresvc})
df5 = pd.DataFrame({'Model': ['Logistic Regression']*len(scoreslr), 'Accuracy': scoreslr})
data = pd.concat([df1, df2, df3, df4, df5])
plt.figure(figsize=(8, 5))
plt.title("The 10 fold cross-validation accuracy score of 5 models")
sns.boxplot(x='Model', y='Accuracy', data=data,hue='Model',width=0.5)
plt.savefig('Figure/KIRC_DifferentModelboxplot.png', dpi=300)
# Train and Test the model
start = time.time()
gbm.fit(X_train, y_train, sample_weight=w)
xg.fit(X_train, y_train, sample_weight=w)
rf.fit(X_train, y_train, sample_weight=w)
svc.fit(X_train, y_train, sample_weight=w)
lr.fit(X_train, y_train, sample_weight=w)
# Predict on test set
y_pred_gbm = gbm.predict(X_test)
y_pred_xg = xg.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_svc = svc.predict(X_test)
y_pred_lr = lr.predict(X_test)
# Model evaluation accuracy
acc_gbm = accuracy_score(y_test, y_pred_gbm)
acc_xg = accuracy_score(y_test, y_pred_xg)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_svc = accuracy_score(y_test, y_pred_svc)
acc_lr = accuracy_score(y_test, y_pred_lr)
# print the cross-validation scores as well as the mean of the scores and plus 95% confidence interval
scores =[scoresgbm,scoresxg,scoresrf,scoresvc,scoreslr]
for i in scores :
      mean_accuracy = np.mean(i) * 100 
      sem = stats.sem(i) * 100  
      confidence_interval = 1.96 * sem
    # print combine
      print(f"The model achieved a 10-fold CV accuracy of {mean_accuracy:.2f}% ± {confidence_interval:.2f}%")
# print the accuracy score of test set as well as the 95% confidence interval
acc = [acc_gbm,acc_xg,acc_rf,acc_svc,acc_lr]
for i in acc :
      i = i * 100
      print(f"The model achieved a test set accuracy of {i:.2f}%")
```


## [Ligthgbm/XGBoost/RandomForest/logistic regression/SVC] 5 models and earlylate prog gene to classify early and late stage for KIRP
```python
# Read data
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# E_L_Stage
# early    189
# late      96
# check the number of early and late stage
KIRP_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRP_GeneExpr = KIRP_Expr[KIRP_ELgene[KIRP_ELgene['ProgStage']=='earlylate']['gene']]
KIRP_Target = KIRP_Expr['E_L_Stage']
# Log2 transformation
KIRP_GeneExpr = np.log2(KIRP_GeneExpr+1)  # !!!!!
# Standardization
scaler = StandardScaler()  # !!!!!
KIRP_GeneExpr = scaler.fit_transform(KIRP_GeneExpr)
# Split data into training and testing sets
y = KIRP_Target
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExpr,y,test_size=0.3, stratify=y,random_state=44) # !!!!!
# SMOTE
sm = SMOTE(random_state=44, sampling_strategy='minority',k_neighbors=40)  # !!!!!
X_train, y_train = sm.fit_resample(X_train, y_train)
# Defult parameters for 5 models
gbm = lgb.LGBMClassifier(random_state=44,boosting_type = 'gbdt',objective = 'binary', n_jobs = 50, metric = 'binary_logloss') # !!!!!
xg = xgb.XGBClassifier(use_label_encoder=True, eval_metric='logloss',random_state=55) # !!!!!
rf = RandomForestClassifier(random_state=44) # !!!!!
svc = SVC(kernel='linear', random_state=44)
lr = LogisticRegression(random_state=44)

# 5-Fold Cross-validation scores
skf = StratifiedKFold(n_splits=10, random_state=44, shuffle=True)
scoresgbm = cross_val_score(gbm, X_train, y_train, cv=skf, scoring='accuracy')
scoresxg = cross_val_score(xg, X_train, y_train, cv=skf, scoring='accuracy')
scoresrf = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy')
scoresvc = cross_val_score(svc, X_train, y_train, cv=skf, scoring='accuracy')
scoreslr = cross_val_score(lr, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scoresgbm),'\n',
      'Cross-validation scores:{}'.format(scoresxg),'\n',
      'Cross-validation scores:{}'.format(scoresrf),'\n',
      'Cross-validation scores:{}'.format(scoresvc),'\n',
      'Cross-validation scores:{}'.format(scoreslr))

# Plotting the boxplot
df1 = pd.DataFrame({'Model': ['Lightgbm']*len(scoresgbm), 'Accuracy': scoresgbm})
df2 = pd.DataFrame({'Model': ['XGBoost']*len(scoresxg), 'Accuracy': scoresxg})
df3 = pd.DataFrame({'Model': ['Random Forest']*len(scoresrf), 'Accuracy': scoresrf})
df4 = pd.DataFrame({'Model': ['SVC']*len(scoresvc), 'Accuracy': scoresvc})
df5 = pd.DataFrame({'Model': ['Logistic Regression']*len(scoreslr), 'Accuracy': scoreslr})
data = pd.concat([df1, df2, df3, df4, df5])
plt.figure(figsize=(8, 5))
plt.title("The 10 fold cross-validation accuracy score of 5 models")
sns.boxplot(x='Model', y='Accuracy', data=data,hue='Model',width=0.5)
plt.savefig('Figure/KIRP_DifferentModelboxplot.png', dpi=300)
# Train and Test the model
start = time.time()
gbm.fit(X_train, y_train)
xg.fit(X_train, y_train)
rf.fit(X_train, y_train)
svc.fit(X_train, y_train)
lr.fit(X_train, y_train)
# Predict on test set
y_pred_gbm = gbm.predict(X_test)
y_pred_xg = xg.predict(X_test)
y_pred_rf = rf.predict(X_test)
y_pred_svc = svc.predict(X_test)
y_pred_lr = lr.predict(X_test)
# Model evaluation accuracy
acc_gbm = accuracy_score(y_test, y_pred_gbm)
acc_xg = accuracy_score(y_test, y_pred_xg)
acc_rf = accuracy_score(y_test, y_pred_rf)
acc_svc = accuracy_score(y_test, y_pred_svc)
acc_lr = accuracy_score(y_test, y_pred_lr)
# print the cross-validation scores as well as the mean of the scores and plus 95% confidence interval
scores =[scoresgbm,scoresxg,scoresrf,scoresvc,scoreslr]
for i in scores :
      mean_accuracy = np.mean(i) * 100 
      sem = stats.sem(i) * 100  
      confidence_interval = 1.96 * sem
    # print combine
      print(f"The model achieved a 10-fold CV accuracy of {mean_accuracy:.2f}% ± {confidence_interval:.2f}%")
# print the accuracy score of test set as well as the 95% confidence interval
acc = [acc_gbm,acc_xg,acc_rf,acc_svc,acc_lr]
for i in acc :
      i = i * 100
      print(f"The model achieved a test set accuracy of {i:.2f}%")
```


# [Tuned model] performance for 5 models
## [Ligthgbm-KIRC] Parameter tuning for KIRC
```python
# Read data
KIRC_ELgene = pd.read_csv('Data/KIRCearlylatelabelCoxProggene005.csv', index_col=0)
KIRC_Eexpr = pd.read_csv('Data/KIRCearlystageExprandClin.csv', index_col=0)
KIRC_Lexpr = pd.read_csv('Data/KIRClatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRC_Eexpr['E_L_Stage'] = 'early'
KIRC_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRC_Expr = pd.concat([KIRC_Eexpr,KIRC_Lexpr],axis=0)
# check the number of early and late stage
KIRC_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRC_GeneExpr = KIRC_Expr[KIRC_ELgene[KIRC_ELgene['ProgStage']=='earlylate']['gene']]
KIRC_Target = KIRC_Expr['E_L_Stage']
# Log2 transformation
KIRC_GeneExpr = np.log2(KIRC_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRC_GeneExpr = scaler.fit_transform(KIRC_GeneExpr)

# Split data into training and testing sets
y = KIRC_Target
X_train, X_test, y_train, y_test = train_test_split(KIRC_GeneExpr,KIRC_Target,test_size=0.3, stratify=y,random_state=44)
w = compute_sample_weight(class_weight='balanced', y=y_train)
# Create a model with default parameters
gbm = lgb.LGBMClassifier(random_state=44,boosting_type = 'gbdt',objective = 'binary', n_jobs = 50, metric = 'binary_logloss')
# Tuning parameters use RandomizedSearchCV
# Create the grid
grid = {
    'n_estimators': list(range(100, 1000,50)),
    'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1],
    'num_leaves': list(range(20, 60,5)),
    'max_depth': list(range(1, 20,2)),
    'subsample': np.linspace(0.1, 1, 10)
}
# Create the random search
random_search = RandomizedSearchCV(
    estimator=gbm, 
    param_distributions=grid, 
    n_iter=100,
    scoring='accuracy',
    cv=3, 
    verbose=1
)
random_search.fit(X_train, y_train)
print('Best parameters found: ', random_search.best_params_, '\n',
      'Accuracy score: ', random_search.best_score_)
y_pred = random_search.best_estimator_.predict(X_test)
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
```

## [Ligthgbm-KIRP] Parameter tuning for KIRP
```python
# Read data
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# E_L_Stage
# early    189
# late      96
# check the number of early and late stage
KIRP_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRP_GeneExpr = KIRP_Expr[KIRP_ELgene[KIRP_ELgene['ProgStage']=='earlylate']['gene']]
KIRP_Target = KIRP_Expr['E_L_Stage']
# Log2 transformation
KIRP_GeneExpr = np.log2(KIRP_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRP_GeneExpr = scaler.fit_transform(KIRP_GeneExpr)
# Split data into training and testing sets
y = KIRP_Target
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExpr,KIRP_Target,test_size=0.3, stratify=y,random_state=44)
# SMOTE
sm = SMOTE(random_state=40, sampling_strategy='minority',k_neighbors=40)
X_train, y_train = sm.fit_resample(X_train, y_train)
# Create a model with default parameters
gbm = lgb.LGBMClassifier(random_state=44,boosting_type = 'gbdt',objective = 'binary', n_jobs = 50, metric = 'binary_logloss')
# Tuning parameters use RandomizedSearchCV
# Create the grid
grid = {
    'n_estimators': list(range(100, 1000,50)),
    'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1],
    'num_leaves': list(range(20, 60,5)),
    'max_depth': list(range(1, 20,2)),
    'subsample': np.linspace(0.1, 1, 10)
}
# Create the random search
random_search = RandomizedSearchCV(
    estimator=gbm, 
    param_distributions=grid, 
    n_iter=100,
    scoring='accuracy',
    cv=3, 
    verbose=1
)
start = time.time()
random_search.fit(X_train, y_train)
print('Best parameters found: ', random_search.best_params_, '\n',
      'Accuracy score: ', random_search.best_score_)
y_pred = random_search.best_estimator_.predict(X_test)
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
end = time.time()
print("Time taken to run:", end - start) 
```

## [XGBoost-KIRC] Parameter tuning for KIRC
```python
# Read data
KIRC_ELgene = pd.read_csv('Data/KIRCearlylatelabelCoxProggene005.csv', index_col=0)
KIRC_Eexpr = pd.read_csv('Data/KIRCearlystageExprandClin.csv', index_col=0)
KIRC_Lexpr = pd.read_csv('Data/KIRClatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRC_Eexpr['E_L_Stage'] = 'early'
KIRC_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRC_Expr = pd.concat([KIRC_Eexpr,KIRC_Lexpr],axis=0)
# check the number of early and late stage
KIRC_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRC_GeneExpr = KIRC_Expr[KIRC_ELgene[KIRC_ELgene['ProgStage']=='earlylate']['gene']]
KIRC_Target = KIRC_Expr['E_L_Stage']
# Log2 transformation
KIRC_GeneExpr = np.log2(KIRC_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRC_GeneExpr = scaler.fit_transform(KIRC_GeneExpr)
# Create a label encoder
le = LabelEncoder()
# Label encoding for target variable
KIRC_Target = le.fit_transform(KIRC_Target)
# Split data into training and testing sets
y = KIRC_Target
X_train, X_test, y_train, y_test = train_test_split(KIRC_GeneExpr,KIRC_Target,test_size=0.3, stratify=y,random_state=44)
w = compute_sample_weight(class_weight='balanced', y=y_train)
# Create a model with default parameters
xgb = XGBClassifier(use_label_encoder=True, eval_metric='logloss')
# Tuning parameters use RandomizedSearchCV
# Create the grid
grid = {
    'n_estimators': list(range(100, 1000,50)),
    'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1],
    'colsample_bytree': np.linspace(0.1, 1, 10),
    'max_depth': list(range(1, 10,2)),
    'subsample': np.linspace(0.1, 1, 10)
}
# Create the random search
random_search = RandomizedSearchCV(
    estimator=xgb, 
    param_distributions=grid, 
    n_iter=100,
    scoring='accuracy',
    cv=3, 
    verbose=1
)
random_search.fit(X_train, y_train)
print('Best parameters found: ', random_search.best_params_, '\n',
      'Accuracy score: ', random_search.best_score_)
# Best parameters found:  {'subsample': 0.2, 'n_estimators': 850, 'max_depth': 9, 'learning_rate': 0.1, 'colsample_bytree': 0.7000000000000001}
# Accuracy score:  0.8590785907859079
y_pred = random_search.best_estimator_.predict(X_test)
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
# Confusion matrix:  
# [[89  8]
# [ 7 55]]
# Accuracy score:  0.9056603773584906
# Recall score:  0.9023112736947123
# Precision score:  0.9000496031746033
# F1 score:  0.901139896373057
# Matthews correlation coefficient: 0.8023576892988286
```

## [XGBoost-KIRP] Parameter tuning for KIRP
```python
# Read data
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# E_L_Stage
# early    189
# late      96
# check the number of early and late stage
KIRP_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRP_GeneExpr = KIRP_Expr[KIRP_ELgene[KIRP_ELgene['ProgStage']=='earlylate']['gene']]
KIRP_Target = KIRP_Expr['E_L_Stage']
# Log2 transformation
KIRP_GeneExpr = np.log2(KIRP_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRP_GeneExpr = scaler.fit_transform(KIRP_GeneExpr)
# Create a label encoder
le = LabelEncoder()
# Label encoding for target variable
KIRP_Target = le.fit_transform(KIRP_Target)
# Split data into training and testing sets
y = KIRP_Target
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExpr,KIRP_Target,test_size=0.3, stratify=y,random_state=44) # KIRP_Target,test_size=0.3 # random_state=44
# SMOTE
sm = SMOTE(random_state=37, sampling_strategy='minority',k_neighbors=40) # random_state=44 # k_neighbors=40
X_train, y_train = sm.fit_resample(X_train, y_train)
# Create a model with default parameters
xgb = XGBClassifier(use_label_encoder=True, eval_metric='logloss',objective='binary:logistic')
# Tuning parameters use RandomizedSearchCV
# Create the grid
grid = {
    'n_estimators': list(range(100, 1000,10)),
    'learning_rate': [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 1],
    'colsample_bytree': np.linspace(0.1, 1, 20),
    'max_depth': list(range(1, 10,1)),
    'subsample': np.linspace(0.1, 1, 20),
    'min_child_weight':list(range(1, 20, 1))
}
# Create the random search
random_search = RandomizedSearchCV(
    estimator=xgb, 
    param_distributions=grid, 
    n_iter=100, 
    scoring='accuracy',
    cv=3, 
    verbose=1
)
start = time.time()
random_search.fit(X_train, y_train)
print('Best parameters found: ', random_search.best_params_, '\n',
      'Accuracy score: ', random_search.best_score_)
end = time.time()
print("Time taken to run:", end - start) 
#Time taken to run: 303.0863857269287 300 times
y_pred = random_search.best_estimator_.predict(X_test)

print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
```

```
>>> print('Best parameters found: ', random_search.best_params_, '\n',
...       'Accuracy score: ', random_search.best_score_)
Best parameters found:  {'subsample': 0.7000000000000001, 'n_estimators': 300, 'min_child_weight': 1, 'max_depth': 2, 'learning_rate': 0.05, 'colsample_bytree': 0.6}
 Accuracy score:  0.837121212121212
>>> end = time.time()
>>> print("Time taken to run:", end - start)
Time taken to run: 50.55003309249878
>>> #Time taken to run: 303.0863857269287 300 times
>>> y_pred = random_search.best_estimator_.predict(X_test)
>>>
>>> print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
Confusion matrix:  [[51  6]
 [ 9 20]]
>>> print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
...       'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
...       'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
...       'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
...       'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
Accuracy score:  0.8255813953488372
 Recall score:  0.7921960072595282
 Precision score:  0.8096153846153846
 F1 score:  0.7995337995337997
 Matthews correlation coefficient: 0.6015592378834806
```


## [RandomForest-KIRC] Parameter tuning for KIRC
```python
# Read data
KIRC_ELgene = pd.read_csv('Data/KIRCearlylatelabelCoxProggene005.csv', index_col=0)
KIRC_Eexpr = pd.read_csv('Data/KIRCearlystageExprandClin.csv', index_col=0)
KIRC_Lexpr = pd.read_csv('Data/KIRClatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRC_Eexpr['E_L_Stage'] = 'early'
KIRC_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRC_Expr = pd.concat([KIRC_Eexpr,KIRC_Lexpr],axis=0)
# check the number of early and late stage
KIRC_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRC_GeneExpr = KIRC_Expr[KIRC_ELgene[KIRC_ELgene['ProgStage']=='earlylate']['gene']]
KIRC_Target = KIRC_Expr['E_L_Stage']
# Log2 transformation
KIRC_GeneExpr = np.log2(KIRC_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRC_GeneExpr = scaler.fit_transform(KIRC_GeneExpr)
# Create a label encoder
le = LabelEncoder()
# Label encoding for target variable
KIRC_Target = le.fit_transform(KIRC_Target)
# Split data into training and testing sets
y = KIRC_Target
X_train, X_test, y_train, y_test = train_test_split(KIRC_GeneExpr,KIRC_Target,test_size=0.3, stratify=y,random_state=44)
w = compute_sample_weight(class_weight='balanced', y=y_train)
# Create a model with default parameters
rf = RandomForestClassifier(random_state=44,class_weight='balanced')
# Tuning parameters use RandomizedSearchCV
# Create the grid
grid = {
    'n_estimators': list(range(100, 1000,10)),
    'max_depth': list(range(1, 20,1)),
    'min_samples_split':list(range(2, 20, 1)),
    'min_samples_leaf': list(range(1, 20, 1)),
    'max_samples': np.linspace(0.1, 1, 20),
    'max_features': np.linspace(0.1, 1, 20)
}
# Create the random search
random_search = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=grid, 
    n_iter=100,
    scoring='accuracy',
    cv=3, 
    verbose=1,
    n_jobs=50
)
random_search.fit(X_train, y_train)
print('Best parameters found: ', random_search.best_params_, '\n',
      'Accuracy score: ', random_search.best_score_)
y_pred = random_search.best_estimator_.predict(X_test)
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
```


```
>>> print('Best parameters found: ', random_search.best_params_, '\n',
...       'Accuracy score: ', random_search.best_score_)
Best parameters found:  {'n_estimators': 710, 'min_samples_split': 9, 'min_samples_leaf': 3, 'max_samples': 0.9052631578947369, 'max_depth': 6}   
 Accuracy score:  0.7723577235772359
>>> y_pred = random_search.best_estimator_.predict(X_test)
>>> print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
Confusion matrix:  [[88  9]
 [10 52]]
>>> print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
...       'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
...       'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
...       'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
...       'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
Accuracy score:  0.8805031446540881
 Recall score:  0.8729630861323578
 Precision score:  0.875209100033456 
 F1 score:  0.8740462789243277
 Matthews correlation coefficient: 0.7481688148898544
```


## [RandomForest-KIRP] Parameter tuning for KIRP
```python
# Read data
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# E_L_Stage
# early    189
# late      96
# check the number of early and late stage
KIRP_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRP_GeneExpr = KIRP_Expr[KIRP_ELgene[KIRP_ELgene['ProgStage']=='earlylate']['gene']]
KIRP_Target = KIRP_Expr['E_L_Stage']
# Log2 transformation
KIRP_GeneExpr = np.log2(KIRP_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRP_GeneExpr = scaler.fit_transform(KIRP_GeneExpr)
# Create a label encoder
le = LabelEncoder()
# Label encoding for target variable
KIRP_Target = le.fit_transform(KIRP_Target)
# Split data into training and testing sets
y = KIRP_Target
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExpr,KIRP_Target,test_size=0.3, stratify=y,random_state=44) # KIRP_Target,test_size=0.3 # random_state=44
# SMOTE
sm = SMOTE(random_state=44, sampling_strategy='minority',k_neighbors=40) # random_state=44 # k_neighbors=40
X_train, y_train = sm.fit_resample(X_train, y_train)
# Create a model with default parameters
rf = RandomForestClassifier(random_state=44)
# Tuning parameters use RandomizedSearchCV
# Create the grid
grid = {
    'n_estimators': list(range(100, 1000,10)),
    'max_depth': list(range(1, 100,1)),
    'min_samples_split':list(range(2, 20, 1)),
    'min_samples_leaf': list(range(1, 20, 1)),
    'max_samples': np.linspace(0.1, 1, 20),
    'max_features': np.linspace(0.1, 1, 20)
}
# Create the random search
random_search = RandomizedSearchCV(
    estimator=rf, 
    param_distributions=grid, 
    n_iter=500,
    scoring='accuracy',
    cv=3, 
    verbose=1,
    n_jobs=50
)
start = time.time()
random_search.fit(X_train, y_train)
print('Best parameters found: ', random_search.best_params_, '\n',
      'Accuracy score: ', random_search.best_score_)
end = time.time()
# Time taken to run: 269.9670960903168 1500times
print("Time taken to run:", end - start) #Time taken to run: 303.0863857269287 300 times
y_pred = random_search.best_estimator_.predict(X_test)
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
```


# [Optimized model] performance for 5 models
## [Ligthgbm-KIRC-Best-model] Best model for KIRC
```python
# Read data
KIRC_ELgene = pd.read_csv('Data/KIRCearlylatelabelCoxProggene005.csv', index_col=0)
KIRC_Eexpr = pd.read_csv('Data/KIRCearlystageExprandClin.csv', index_col=0)
KIRC_Lexpr = pd.read_csv('Data/KIRClatestageExprandClin.csv', index_col=0)
GENEcompare = pd.read_csv('Data/probeMap_gencode.v23.annotation.gene.probemap', sep='\t')
# Using regular expression to extract gene ensemble id with out version .1.2
GENEcompare['id'] = GENEcompare['id'].str.split('.').str[0]
# add label for early and late stage
KIRC_Eexpr['E_L_Stage'] = 'early'
KIRC_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRC_Expr = pd.concat([KIRC_Eexpr,KIRC_Lexpr],axis=0)
# check the number of early and late stage
KIRC_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage 
KIRC_GeneExpr = KIRC_Expr[KIRC_ELgene[KIRC_ELgene['ProgStage']=='earlylate']['gene']]
# Change colname(ensemble id) to gene name
KIRC_GeneExpr.columns = GENEcompare[GENEcompare['id'].isin(KIRC_GeneExpr.columns)]['gene'].values
KIRC_Target = KIRC_Expr['E_L_Stage']
# Log2 transformation
KIRC_GeneExpr = np.log2(KIRC_GeneExpr+1) # !!!!!
# Standardization 
scaler = StandardScaler() # !!!!!
KIRC_GeneExpr = scaler.fit_transform(KIRC_GeneExpr)
# Split data into training and testing sets
le = LabelEncoder()
y = le.fit_transform(KIRC_Target)
X_train, X_test, y_train, y_test = train_test_split(KIRC_GeneExpr,y,test_size=0.3, stratify=y,random_state=44) # !!!!!
w = compute_sample_weight(class_weight='balanced', y=y_train) 
# Best parameters
params = {'subsample': 0.7000000000000001, 'num_leaves': 35, 'n_estimators': 500, 'max_depth': 1, 'learning_rate': 0.5} # !!!!!
# Create a model with default parameters
gbm = lgb.LGBMClassifier(**params,random_state=44,probability=True) # !!!!!
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=5, random_state=37, shuffle=True)
# Initialize the list to store the scores
scoresKIRC = cross_val_score(gbm, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scoresKIRC))
# Plotting the cross-validation scores
# Train and Test the model
start = time.time()
gbm.fit(X_train, y_train, sample_weight=w)
# Predict on test set
y_pred = gbm.predict(X_test)
# Model evaluation
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
end = time.time()
# Plotting the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
plt.title("Confusion Matrix of KIRC")
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('Figure/KIRC_GBM_confusion_matrix.png', dpi=300)
plt.close()
# Plotting the ROC curve
y_pred_proba = gbm.predict_proba(X_test)[::, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label="auc="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange', label='Baseline')
plt.legend(loc=4)
plt.savefig('Figure/KIRC_GBM_ROC_curve.png', dpi=300)
plt.close()
# Plotting the precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba, pos_label=1)
plt.figure(figsize=(5, 5))
plt.plot(recall, precision, label="auc="+str(auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('Figure/KIRC_GBM_precision_recall_curve.png', dpi=300)
```

```
5-Fold Cross-validation scores:[0.83783784 0.85135135 0.7972973  0.81081081 0.87671233]
Best parameters found:  {'subsample': 0.7000000000000001, 'num_leaves': 35, 'n_estimators': 500, 'max_depth': 1, 'learning_rate': 0.5} 
Confusion matrix:  
[[90  7]
 [ 5 57]]
Accuracy score:  0.9245283018867925 
Recall score:  0.9235949451280345
Precision score:  0.9189967105263157
F1 score:  0.9211309523809523
Matthews correlation coefficient: 0.8425791086995256
```


## [Ligthgbm-KIRP-Best-model] Best model for KIRP
```python
# Read data
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# E_L_Stage
# early    189
# late      96
# check the number of early and late stage
KIRP_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRP_GeneExpr = KIRP_Expr[KIRP_ELgene[KIRP_ELgene['ProgStage']=='earlylate']['gene']]
KIRP_Target = KIRP_Expr['E_L_Stage']
# Log2 transformation
KIRP_GeneExpr = np.log2(KIRP_GeneExpr+1)  # !!!!!
# Standardization
scaler = StandardScaler()  # !!!!!
KIRP_GeneExpr = scaler.fit_transform(KIRP_GeneExpr)
# Split data into training and testing sets
le = LabelEncoder()
y = le.fit_transform(KIRP_Target)
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExpr,y,test_size=0.3, stratify=y,random_state=44) # !!!!!
# SMOTE
sm = SMOTE(random_state=44, sampling_strategy='minority',k_neighbors=40)  # !!!!!
X_train, y_train = sm.fit_resample(X_train, y_train)
# Best parameters
params = {'subsample': 0.8, 'num_leaves': 55, 'n_estimators': 150, 'max_depth': 5, 'learning_rate': 1} # !!!!!
# Create a model with default parameters
gbm = lgb.LGBMClassifier(**params,boosting_type = 'gbdt',objective = 'binary',metric = 'binary_logloss',random_state=44,probability=True) # !!!!!
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=5, random_state=44, shuffle=True)  # !!!!!
# Initialize the list to store the scores
scoresKIRP = cross_val_score(gbm, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scoresKIRP))
# Train model
gbm.fit(X_train, y_train)
# Predict on test set
y_pred = gbm.predict(X_test)
# random_search.fit(X_train, y_train)
print('Best parameters found: ', random_search.best_params_, '\n',
      'Accuracy score: ', random_search.best_score_)
#y_pred = random_search.best_estimator_.predict(X_test)
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))

print('Time taken to run:', time_end - time_start)
# Plotting the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
plt.title("Confusion Matrix of KIRP")
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('Figure/KIRP_GBM_confusion_matrix.png', dpi=300)
plt.close()
# Plotting the ROC curve
y_pred_proba = gbm.predict_proba(X_test)[::, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label="auc="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange', label='Baseline')
plt.legend(loc=4)
plt.savefig('Figure/KIRP_GBM_ROC_curve.png', dpi=300)
plt.close()
# Plotting the precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba, pos_label=1)
plt.figure(figsize=(5, 5))
plt.plot(recall, precision, label="auc="+str(auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('Figure/KIRP_GBM_precision_recall_curve.png', dpi=300)
```
```
Best parameters found:  {'subsample': 0.8, 'num_leaves': 20, 'n_estimators': 220, 'max_depth': 10, 'learning_rate': 0.1, 'colsample_bytree': 0.1}
Confusion matrix:  
[[53  4]
 [ 8 21]]
Accuracy score:  0.8604651162790697
Recall score:  0.8269812462189958
Precision score:  0.8544262295081967
F1 score:  0.8380414312617701
Matthews correlation coefficient: 0.6808545519192479
Time taken to run: 0.008035659790039062
```


## [LogisticRegression-KIRC-Best-model] Best model for KIRC
```python
# Read data
KIRC_ELgene = pd.read_csv('Data/KIRCearlylatelabelCoxProggene005.csv', index_col=0)
KIRC_Eexpr = pd.read_csv('Data/KIRCearlystageExprandClin.csv', index_col=0)
KIRC_Lexpr = pd.read_csv('Data/KIRClatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRC_Eexpr['E_L_Stage'] = 'early'
KIRC_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRC_Expr = pd.concat([KIRC_Eexpr,KIRC_Lexpr],axis=0)
# check the number of early and late stage
KIRC_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRC_GeneExpr = KIRC_Expr[KIRC_ELgene[KIRC_ELgene['ProgStage']=='earlylate']['gene']]
KIRC_Target = KIRC_Expr['E_L_Stage']
# Log2 transformation
KIRC_GeneExpr = np.log2(KIRC_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRC_GeneExpr = scaler.fit_transform(KIRC_GeneExpr)
# Split data into training and testing sets
le = LabelEncoder()
y = le.fit_transform(KIRC_Target)
X_train, X_test, y_train, y_test = train_test_split(KIRC_GeneExpr,y,test_size=0.3, stratify=y,random_state=44)
w = compute_sample_weight(class_weight='balanced', y=y_train)
# Create a model with default parameters
logreg = LogisticRegression(max_iter=600,penalty='l2',C=0.8,random_state=44)
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=5, random_state=37, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(logreg, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# Cross-validation scores:[0.89189189 0.93243243 0.94594595 0.95945946 0.95890411]

# Train and Test the model
start = time.time()
logreg.fit(X_train, y_train, sample_weight=w)
# Predict on test set
y_pred = logreg.predict(X_test)
# Model evaluation
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
end = time.time()
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
plt.title("Confusion Matrix of KIRC")
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5, 1.5], ['early', 'late'])
plt.yticks([0.5, 1.5], ['early', 'late'])
plt.savefig('Figure/KIRC_LR_confusion_matrix.png', dpi=300)
plt.close()
# Plotting the ROC curve
y_pred_proba = logreg.predict_proba(X_test)[::, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label="auc="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange', label='Baseline')
plt.legend(loc=4)
plt.savefig('Figure/KIRC_LR_ROC_curve.png', dpi=300)
plt.close()
# Plotting the precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba, pos_label=1)
plt.figure(figsize=(5, 5))
plt.plot(recall, precision, label="auc="+str(auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('Figure/KIRC_LR_precision_recall_curve.png', dpi=300)
```
```
# Confusion matrix: 
# [[92  5]
# [ 2 60]]
# Accuracy score:  0.9559748427672956
# Recall score:  0.9580977718656468
# Precision score:  0.9509001636661212
# F1 score:  0.9541163375520468
# Matthews correlation coefficient: 0.9089694391107009
```

## [LogisticRegression-KIRP-Best-model] Best model for KIRP
```python
# Read data
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# E_L_Stage
# early    189
# late      96
# check the number of early and late stage
KIRP_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRP_GeneExpr = KIRP_Expr[KIRP_ELgene[KIRP_ELgene['ProgStage']=='earlylate']['gene']]
KIRP_Target = KIRP_Expr['E_L_Stage']
# Log2 transformation
KIRP_GeneExpr = np.log2(KIRP_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRP_GeneExpr = scaler.fit_transform(KIRP_GeneExpr)
# Split data into training and testing sets
le = LabelEncoder()
y = le.fit_transform(KIRP_Target)
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExpr,y,test_size=0.3, stratify=y,random_state=44)
# SMOTE
sm = SMOTE(random_state=44, sampling_strategy='minority',k_neighbors=40)
X_train, y_train = sm.fit_resample(X_train, y_train)
# Create a model with default parameters
logreg = LogisticRegression(random_state=44, max_iter=1000,C = 0.8, penalty = 'l2')
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=10, random_state=44, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(lr, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# Cross-validation scores:[0.88888889 0.96296296 0.92592593 0.85185185 0.80769231 1. 0.92307692 0.88461538 0.92307692 0.96153846]
# Train and Test the model
start = time.time()
lr.fit(X_train, y_train)
# Predict on test set
y_pred = lr.predict(X_test)
# Model evaluation
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
end = time.time()
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
plt.title("Confusion Matrix of KIRP")
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5, 1.5], ['early', 'late'])
plt.yticks([0.5, 1.5], ['early', 'late'])
plt.savefig('Figure/KIRP_LR_confusion_matrix.png', dpi=300)
plt.close()
# Plotting the ROC curve
y_pred_proba = lr.predict_proba(X_test)[::, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label="auc="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange', label='Baseline')
plt.legend(loc=4)
plt.savefig('Figure/KIRP_LR_ROC_curve.png', dpi=300)
plt.close()
# Plotting the precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba, pos_label=1)
plt.figure(figsize=(5, 5))
plt.plot(recall, precision, label="auc="+str(auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('Figure/KIRP_LR_precision_recall_curve.png', dpi=300)
plt.close()


# Confusion matrix:  [[52  5]
# [ 7 22]]
# Accuracy score:  0.8604651162790697
# Recall score:  0.8354506957047791
# Precision score:  0.8480853735091023
# F1 score:  0.8411330049261083
# Matthews correlation coefficient: 0.6834192877239749
```

## [SVC-KIRC-Best-model] Best model for KIRC
```python
# Read data
KIRC_ELgene = pd.read_csv('Data/KIRCearlylatelabelCoxProggene005.csv', index_col=0)
KIRC_Eexpr = pd.read_csv('Data/KIRCearlystageExprandClin.csv', index_col=0)
KIRC_Lexpr = pd.read_csv('Data/KIRClatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRC_Eexpr['E_L_Stage'] = 'early'
KIRC_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRC_Expr = pd.concat([KIRC_Eexpr,KIRC_Lexpr],axis=0)
# check the number of early and late stage
KIRC_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRC_GeneExpr = KIRC_Expr[KIRC_ELgene[KIRC_ELgene['ProgStage']=='earlylate']['gene']]
KIRC_Target = KIRC_Expr['E_L_Stage']
# Log2 transformation
KIRC_GeneExpr = np.log2(KIRC_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRC_GeneExpr = scaler.fit_transform(KIRC_GeneExpr)
# Split data into training and testing sets
le = LabelEncoder()
y = le.fit_transform(KIRC_Target)
X_train, X_test, y_train, y_test  = train_test_split(KIRC_GeneExpr,y,test_size=0.3, stratify=y,random_state=44)
w = compute_sample_weight(class_weight='balanced', y=y_train)
# Create a model with default parameters
svc = SVC(kernel='linear', random_state=44,probability=True)
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=10, random_state=37, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(svc, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# Cross-validation scores:[0.98648649 0.97297297 0.94594595 0.98648649 0.94520548]
# Train and Test the model
start = time.time()
svc.fit(X_train, y_train, sample_weight=w)
# Predict on test set
y_pred = svc.predict(X_test)
# Model evaluation
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
end = time.time()

# plot_confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
plt.title("Confusion Matrix of KIRC")
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5, 1.5], ['early', 'late'])
plt.yticks([0.5, 1.5], ['early', 'late'])
plt.savefig('Figure/KIRC_SVC_confusion_matrix.png', dpi=300)
plt.close()

# Plotting the ROC curve
y_pred_proba = svc.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label="auc="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange', label='Baseline')
plt.legend(loc=4)
plt.savefig('Figure/KIRC_SVC_ROC_curve.png', dpi=300)
plt.close()
# Plotting the precision-recall curve
y_pred_proba = svc.decision_function(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba, pos_label=1)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(5, 5))
plt.plot(recall, precision, label="auc="+str(auc), color='#3776ab')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('Figure/KIRC_SVC_precision_recall_curve.png', dpi=300)
plt.close()
# Confusion matrix:  
# [[92  5]
# [ 4 58]]
# Accuracy score:  0.9433962264150944
# Recall score:  0.9419687396075822
# Precision score:  0.939484126984127
# F1 score:  0.9406839378238342
# Matthews correlation coefficient: 0.8814493648093764
```

## [SVC-KIRP-Best-model] Best model for KIRP
```python
# Read data
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# E_L_Stage
# early    189
# late      96
# check the number of early and late stage
KIRP_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRP_GeneExpr = KIRP_Expr[KIRP_ELgene[KIRP_ELgene['ProgStage']=='earlylate']['gene']]
KIRP_Target = KIRP_Expr['E_L_Stage']
# Log2 transformation
KIRP_GeneExpr = np.log2(KIRP_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRP_GeneExpr = scaler.fit_transform(KIRP_GeneExpr)
# Split data into training and testing sets
le = LabelEncoder()
y = le.fit_transform(KIRP_Target)
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExpr,y,test_size=0.3, stratify=y,random_state=44)
# SMOTE
sm = SMOTE(random_state=44, sampling_strategy='minority',k_neighbors=40)
X_train, y_train = sm.fit_resample(X_train, y_train)
# Create a model with default parameters
svc = SVC(kernel='linear', random_state=44,probability=True)
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=10, random_state=37, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(svc, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# Cross-validation scores:[0.88888889 0.81481481 0.92592593 1.0.96153846 0.92307692 0.96153846 0.84615385 0.96153846 0.92307692]

# Train and Test the model
start = time.time()
svc.fit(X_train, y_train)
# Predict on test set
y_pred = svc.predict(X_test)

# Model evaluation
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
end = time.time()

# plot_confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
plt.title("Confusion Matrix of KIRP")
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5, 1.5], ['early', 'late'])
plt.yticks([0.5, 1.5], ['early', 'late'])
plt.savefig('Figure/KIRP_SVC_confusion_matrix.png', dpi=300)
plt.close()

# Plotting the ROC curve
y_pred_proba = svc.decision_function(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label="auc="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange', label='Baseline')
plt.legend(loc=4)
plt.savefig('Figure/KIRP_SVC_ROC_curve.png', dpi=300)
plt.close()
# Plotting the precision-recall curve
y_pred_proba = svc.decision_function(X_test)
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba, pos_label=1)
plt.figure(figsize=(5, 5))
plt.plot(recall, precision, label="auc="+str(auc), color='#3776ab')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('Figure/KIRP_SVC_precision_recall_curve.png', dpi=300)
plt.close()
```
```
# Confusion matrix:  
# [[49  8]
# [ 8 21]]
# Accuracy score:  0.813953488372093
# Recall score:  0.7918935269207501
# Precision score:  0.7918935269207501
# F1 score:  0.7918935269207502
# Matthews correlation coefficient: 0.5837870538415003
```

## [XGBoost-KIRC-Best-model] Best model for KIRC
```python
# Read data
KIRC_ELgene = pd.read_csv('Data/KIRCearlylatelabelCoxProggene005.csv', index_col=0)
KIRC_Eexpr = pd.read_csv('Data/KIRCearlystageExprandClin.csv', index_col=0)
KIRC_Lexpr = pd.read_csv('Data/KIRClatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRC_Eexpr['E_L_Stage'] = 'early'
KIRC_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRC_Expr = pd.concat([KIRC_Eexpr,KIRC_Lexpr],axis=0)
# check the number of early and late stage
KIRC_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRC_GeneExpr = KIRC_Expr[KIRC_ELgene[KIRC_ELgene['ProgStage']=='earlylate']['gene']]
KIRC_Target = KIRC_Expr['E_L_Stage']
# Log2 transformation
KIRC_GeneExpr = np.log2(KIRC_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRC_GeneExpr = scaler.fit_transform(KIRC_GeneExpr)
# Create a label encoder
le = LabelEncoder()
# Label encoding for target variable
y = le.fit_transform(KIRC_Target)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(KIRC_GeneExpr,y,test_size=0.3, stratify=y,random_state=44)
# Create a model with best parameters
params = {'subsample': 0.2, 'n_estimators': 850, 'max_depth': 9, 'learning_rate': 0.1, 'colsample_bytree': 0.7000000000000001}
xgb = XGBClassifier(**params,use_label_encoder=True, eval_metric='logloss')
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=10, random_state=44, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(xgb, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# Cross-validation scores:[0.86486486 0.86486486 0.81081081 0.83783784 0.83783784 0.86486486 0.91891892 0.83783784 0.83783784 0.91666667]
xgb.fit(X_train, y_train)
# Predict on test set
y_pred = xgb.predict(X_test)
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))

plot_metrics(xgb, X_test, y_test, y_pred, 'KIRC', 'XGB')
```

```
Best parameters found:  {'subsample': 0.2, 'n_estimators': 850, 'max_depth': 9, 'learning_rate': 0.1, 'colsample_bytree': 0.7000000000000001}
# Confusion matrix:  
# [[89  8]
# [ 7 55]]
# Accuracy score:  0.9056603773584906
# Recall score:  0.9023112736947123
# Precision score:  0.9000496031746033
# F1 score:  0.901139896373057
# Matthews correlation coefficient: 0.8023576892988286
```


## [XGBoost-KIRP-Best-model] Best model for KIRP
```python
# Read data
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# check the number of early and late stage
KIRP_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRP_GeneExpr = KIRP_Expr[KIRP_ELgene[KIRP_ELgene['ProgStage']=='earlylate']['gene']]
KIRP_Target = KIRP_Expr['E_L_Stage']
# Log2 transformation
KIRP_GeneExpr = np.log2(KIRP_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRP_GeneExpr = scaler.fit_transform(KIRP_GeneExpr)
# Create a label encoder
le = LabelEncoder()
# Label encoding for target variable
y = le.fit_transform(KIRP_Target)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExpr,y,test_size=0.3, stratify=y,random_state=44)
# SMOTE
sm = SMOTE(random_state=44, sampling_strategy='minority',k_neighbors=40) # random_state=44 # k_neighbors=40
X_train, y_train = sm.fit_resample(X_train, y_train)
# Create a model with default parameters
params = {'subsample': 0.7000000000000001, 'n_estimators': 500, 'min_child_weight': 1, 'max_depth': 2, 'learning_rate': 0.05, 'colsample_bytree': 0.6}
xgb = XGBClassifier(**params,use_label_encoder=True, eval_metric='logloss',probability=True)
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=10, random_state=44, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(xgb, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# Cross-validation scores:[0.85185185 0.88888889 0.88888889 0.85185185 0.80769231 0.96153846 0.96153846 0.88461538 0.84615385 0.80769231]
xgb.fit(X_train, y_train)
# Predict on test set
y_pred = xgb.predict(X_test)
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))

plot_metrics(xgb, X_test, y_test, y_pred, 'KIRP', 'XGB')
```
```
# Best parameters found:  {'subsample': 0.7000000000000001, 'n_estimators': 500, 'min_child_weight': 1, 'max_depth': 2, 'learning_rate': 0.05, 'colsample_bytree': 0.6} 
# Confusion matrix: 
# [[52  5]
# [ 8 21]]
# Accuracy score:  0.8488372093023255
# Recall score:  0.8182093163944344
# Precision score:  0.8371794871794872
# F1 score:  0.8262626262626263
# Matthews correlation coefficient: 0.6551142010904986
```

## [RandomForest-KIRC-Best-model] Best model for KIRC
```python
# Read data
KIRC_ELgene = pd.read_csv('Data/KIRCearlylatelabelCoxProggene005.csv', index_col=0)
KIRC_Eexpr = pd.read_csv('Data/KIRCearlystageExprandClin.csv', index_col=0)
KIRC_Lexpr = pd.read_csv('Data/KIRClatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRC_Eexpr['E_L_Stage'] = 'early'
KIRC_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRC_Expr = pd.concat([KIRC_Eexpr,KIRC_Lexpr],axis=0)
# check the number of early and late stage
KIRC_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRC_GeneExpr = KIRC_Expr[KIRC_ELgene[KIRC_ELgene['ProgStage']=='earlylate']['gene']]
KIRC_Target = KIRC_Expr['E_L_Stage']
# Log2 transformation
KIRC_GeneExpr = np.log2(KIRC_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRC_GeneExpr = scaler.fit_transform(KIRC_GeneExpr)
# Create a label encoder
le = LabelEncoder()
# Label encoding for target variable
y = le.fit_transform(KIRC_Target)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(KIRC_GeneExpr,y,test_size=0.3, stratify=y,random_state=44)
# Create a model with default parameters
params = {'n_estimators': 710, 'min_samples_split': 9, 'min_samples_leaf': 3, 'max_samples': 0.9052631578947369, 'max_depth': 6} 
rf = RandomForestClassifier(**params,random_state=44,class_weight='balanced')
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=10, random_state=44, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# Cross-validation scores:[0.72972973 0.72972973 0.83783784 0.83783784 0.75675676 0.7027027 0.72972973 0.72972973 0.81081081 0.86111111]
# fit model
rf.fit(X_train, y_train)
# Predict on test set
y_pred = rf.predict(X_test)
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))

plot_metrics(rf, X_test, y_test, y_pred, 'KIRC', 'RF')
```
```
Best parameters found:  {'n_estimators': 710, 'min_samples_split': 9, 'min_samples_leaf': 3, 'max_samples': 0.9052631578947369, 'max_depth': 6} 
# Confusion matrix: 
# [[88  9]
# [10 52]]
# Accuracy score:  0.8805031446540881
# Recall score:  0.8729630861323578
# Precision score:  0.875209100033456 
# F1 score:  0.8740462789243277
# Matthews correlation coefficient: 0.7481688148898544
```

## [RandomForest-KIRP-Best-model] Best model for KIRP
```python
# Read data
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# E_L_Stage
# early    189
# late      96
# check the number of early and late stage
KIRP_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRP_GeneExpr = KIRP_Expr[KIRP_ELgene[KIRP_ELgene['ProgStage']=='earlylate']['gene']]
KIRP_Target = KIRP_Expr['E_L_Stage']
# Log2 transformation
KIRP_GeneExpr = np.log2(KIRP_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRP_GeneExpr = scaler.fit_transform(KIRP_GeneExpr)
# Create a label encoder
le = LabelEncoder()
# Label encoding for target variable
y = le.fit_transform(KIRP_Target)
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExpr,y,test_size=0.3, stratify=y,random_state=44) # KIRP_Target,test_size=0.3 # random_state=44
# SMOTE
sm = SMOTE(random_state=44, sampling_strategy='minority',k_neighbors=40) # random_state=44 # k_neighbors=40
X_train, y_train = sm.fit_resample(X_train, y_train)
# Create a model with default parameters
params = {'n_estimators': 110, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_samples': 0.9052631578947369, 'max_features': 0.7157894736842105, 'max_depth': 33}
rf = RandomForestClassifier(**params,random_state=44)
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=10, random_state=44, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# Cross-validation scores:[0.85185185 0.88888889 0.88888889 0.85185185 0.80769231 0.96153846 0.96153846 0.88461538 0.84615385 0.80769231]
# fit model
rf.fit(X_train, y_train)
# Predict on test set
y_pred = rf.predict(X_test)
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))

plot_metrics(rf, X_test, y_test, y_pred, 'KIRP', 'RF')
```
```
# Best parameters found:  {'n_estimators': 110, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_samples': 0.9052631578947369, 'max_features': 0.7157894736842105, 'max_depth': 33}
# Confusion matrix:  
# [[50  7]
# [ 9 20]]
# Accuracy score:  0.813953488372093
# Recall score:  0.7834240774349668
# Precision score:  0.7940991839296924
# F1 score:  0.7881773399014778 
# Matthews correlation coefficient: 0.5774245920625468
```
## `plot_metrics` function : plot confusion matrix, ROC curve and precision-recall curve for non-linear models
```python
def plot_metrics(model, X_test, y_test, y_pred, cancer_type, model_name):
    # Plotting the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 5))
    plt.title(f"Confusion Matrix of {cancer_type}")
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks([0.5, 1.5], ['early', 'late'])
    plt.yticks([0.5, 1.5], ['early', 'late'])
    plt.savefig(f'Figure/{cancer_type}_{model_name}_confusion_matrix.png', dpi=300)
    plt.close()
    # Plotting the ROC curve
    y_pred_proba = model.predict_proba(X_test)[::, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
    auc = roc_auc_score(y_test, y_pred_proba)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label="auc="+str(auc))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange', label='Baseline')
    plt.legend(loc=4)
    plt.savefig(f'Figure/{cancer_type}_{model_name}_ROC_curve.png', dpi=300)
    plt.close()
    # Plotting the precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba, pos_label=1)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label="auc="+str(auc))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.savefig(f'Figure/{cancer_type}_{model_name}_precision_recall_curve.png', dpi=300)
    plt.close()
```


## `MultimodelROC` function : plot confusion matrix, ROC curve and precision-recall curve for non-linear models
```python
classifiers = {
    "Random Forest": rf,
    "XGBoost": xgb,
    "LightGBM": gbm,
    "SVC": svc,
    "Logistic Regression":logreg
}
# Run classifiers
fig, ax_roc = plt.subplots(figsize=(5, 5))
for name, clf in classifiers.items():
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      y_pred_proba = clf.predict_proba(X_test)[::, 1]
      fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
      auc = roc_auc_score(y_test, y_pred_proba)
      plt.plot(fpr, tpr, label=name+" auc={:.2f}".format(auc))
      plt.legend(loc=4)
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title("ROC curves of KIRC")

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange', label='Baseline')      
plt.savefig('Figure/KIRC_Multimodel_ROC_curve.png', dpi=300)
plt.savefig('Figure/KIRC_Multimodel_ROC_curve.svg', dpi=300)
plt.close()
```
```python
classifiers = {
    "Random Forest": rf,
    "XGBoost": xgb,
    "LightGBM": gbm,
    "SVC": svc,
    "Logistic Regression":logreg

}
# Run classifiers
fig, ax_roc = plt.subplots(figsize=(5, 5))
for name, clf in classifiers.items():
      clf.fit(X_train, y_train)
      y_pred = clf.predict(X_test)
      y_pred_proba = clf.predict_proba(X_test)[::, 1]
      fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
      auc = roc_auc_score(y_test, y_pred_proba)
      plt.plot(fpr, tpr, label=name+" auc={:.2f}".format(auc))
      plt.legend(loc=4)
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title("ROC curves of KIRP")

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange', label='Baseline')      
plt.savefig('Figure/KIRP_Multimodel_ROC_curve.png', dpi=300)
plt.savefig('Figure/KIRP_Multimodel_ROC_curve.svg', dpi=300)
plt.close()
```

# END