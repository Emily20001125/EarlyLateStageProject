# Machine Learning for Stage Classification
Author: Huang,Shu-Jing
Date: 2023-11-28

## environment
```shell
conda activate MachineLearning
cd /home/emily2835/stage_project_git3/StagecClassification_MachineLearning
```

## Import packages for machine learning
```python
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import compute_sample_weight
```

# Original Test

## [Ligthgbm-KIRC] Using earlylate prog gene to classify early and late stage for Kidney cancer
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
gbm = lgb.LGBMClassifier(n_estimators=500)

# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=5, random_state=37, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(gbm, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# Cross-validation scores:[0.81081081 0.85135135 0.82432432 0.68918919 0.82191781]

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

# Accuracy score:  0.8867924528301887 
# Recall score:  0.8722979714000665
# Precision score:  0.8881761442441054
# F1 score:  0.8787288135593219
# Matthews correlation coefficient: 0.760308334948342

print("Time taken to run:", end - start) 
# Time taken to run: 2.809197187423706
```


## [Ligthgbm-KIRP] Using earlylate prog gene to classify early and late stage for Kidney cancer
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
gbm = lgb.LGBMClassifier(boosting_type = 'gbdt',objective = 'binary',n_jobs = 100, metric = 'binary_logloss',random_state = 44)

# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=5, random_state=11, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(gbm, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# Cross-validation scores:[0.79245283 0.88679245 0.81132075 0.81132075 0.84615385]


# Train and Test the model
start = time.time()
gbm.fit(X_train, y_train)
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

# Confusion matrix:  
#  [[52  5]
#  [ 8 21]]
# Accuracy score:  0.8488372093023255 
# Recall score:  0.8182093163944344
# Precision score:  0.8371794871794872
# F1 score:  0.8262626262626263
# Matthews correlation coefficient: 0.6551142010904986

print("Time taken to run:", end - start) 
# Time taken to run: 3.19413161277771
```


## [XGBoost-KIRC] Using earlylate prog gene to classify early and late stage for Kidney cancer
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
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=5, random_state=37, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(xgb, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# Cross-validation scores:[0.78378378 0.78378378 0.81081081 0.74324324 0.73972603]
# Train and Test the model
start = time.time()
xgb.fit(X_train, y_train, sample_weight=w)
# Predict on test set
y_pred = xgb.predict(X_test)
# Model evaluation
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
end = time.time()
```

## [XGBoost-KIRP] Using earlylate prog gene to classify early and late stage for Kidney cancer
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
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExpr,KIRP_Target,test_size=0.3, stratify=y,random_state=44)
# SMOTE
sm = SMOTE(random_state=40, sampling_strategy='minority',k_neighbors=40)
X_train, y_train = sm.fit_resample(X_train, y_train)
# Create a model with default parameters
xgb = XGBClassifier(use_label_encoder=True, eval_metric='logloss')
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=5, random_state=11, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(xgb, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))

# Train and Test the model
start = time.time()
xgb.fit(X_train, y_train)
# Predict on test set
y_pred = xgb.predict(X_test)
# Model evaluation
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
end = time.time()
```

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


## [Ligthgbm-KIRC-Best-model] Parameter tuning for KIRC
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
X_train, X_test, y_train, y_test = train_test_split(KIRC_GeneExpr,KIRC_Target,test_size=0.3, stratify=y,random_state=44) # !!!!!
w = compute_sample_weight(class_weight='balanced', y=y_train) 
# Best parameters
params = {'subsample': 0.7000000000000001, 'num_leaves': 35, 'n_estimators': 500, 'max_depth': 1, 'learning_rate': 0.5} # !!!!!
# Create a model with default parameters
gbm = lgb.LGBMClassifier(**params,random_state=44) # !!!!!
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=5, random_state=37, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(gbm, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
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


## [Ligthgbm-KIRP-Best-model] Parameter tuning for KIRP
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
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExpr,KIRP_Target,test_size=0.3, stratify=y,random_state=44) # !!!!!
# SMOTE
sm = SMOTE(random_state=44, sampling_strategy='minority',k_neighbors=40)  # !!!!!
X_train, y_train = sm.fit_resample(X_train, y_train)
# Best parameters
params = {'subsample': 0.8, 'num_leaves': 55, 'n_estimators': 150, 'max_depth': 5, 'learning_rate': 1} # !!!!!
# Create a model with default parameters
gbm = lgb.LGBMClassifier(**params,boosting_type = 'gbdt',objective = 'binary',metric = 'binary_logloss',random_state=44) # !!!!!
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=5, random_state=44, shuffle=True)  # !!!!!
# Initialize the list to store the scores
scores = cross_val_score(gbm, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
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