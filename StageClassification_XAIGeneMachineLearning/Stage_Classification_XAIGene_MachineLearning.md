# Stage classification using XAIGene and Machine Learning
Author: Huang,Shu-Jing
Date: 2024-01-11

## environment
```shell
conda activate MachineLearning
cd /home/emily2835/EarlyLateStageProject/StageClassification_XAIGeneMachineLearning
```

## Import packages
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
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
import scipy.stats as stats
```

## KIRC - Logistic regression
```python
KIRC_ELgene = pd.read_csv('Data/KIRCearlylatelabelCoxProggene005.csv', index_col=0)
KIRC_Eexpr = pd.read_csv('Data/KIRCearlystageExprandClin.csv', index_col=0)
KIRC_Lexpr = pd.read_csv('Data/KIRClatestageExprandClin.csv', index_col=0)
GENEcompare = pd.read_csv('Data/probeMap_gencode.v23.annotation.gene.probemap', sep='\t')
KIRCXAI5ModelSHAPRank = pd.read_csv('Data/KIRCXAI5ModelSHAPRank.csv', index_col=0)
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
KIRC_GeneExpr = KIRC_Expr[KIRCXAI5ModelSHAPRank['Gene ensemble id']]
# Change colname(ensemble id) to gene name
KIRC_GeneExpr.columns = GENEcompare[GENEcompare['id'].isin(KIRC_GeneExpr.columns)]['gene'].values
KIRC_Target = KIRC_Expr['E_L_Stage']
# Log2 transformation
KIRC_GeneExpr = np.log2(KIRC_GeneExpr+1) # !!!!!
# Standardization 
scaler = StandardScaler() # !!!!!
KIRC_GeneExpr = scaler.fit_transform(KIRC_GeneExpr)
# Split train and test data
le = LabelEncoder()
y = le.fit_transform(KIRC_Target)
X_train, X_test, y_train, y_test = train_test_split(KIRC_GeneExpr, y, test_size=0.2, random_state=44)
w = compute_sample_weight(class_weight='balanced', y=y_train)
# Logistic regression
logreg = LogisticRegression(random_state=44, max_iter=1000,C = 0.8, penalty = 'l2')
skf = StratifiedKFold(n_splits=10, random_state=37, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(logreg, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
logreg.fit(X_train, y_train,sample_weight=w)
y_pred = logreg.predict(X_test)
# print the cross-validation scores as well as the mean of the scores and plus 95% confidence interval
mean_accuracy = np.mean(scores) * 100 
sem = stats.sem(scores) * 100  
confidence_interval = 1.96 * sem
print(f"The model achieved a 10-fold CV accuracy of {mean_accuracy:.2f}% ± {confidence_interval:.2f}%")
# Model evaluation
start = time.time()
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
plt.title("Confusion Matrix of KIRC Logistic Regression")
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5, 1.5], ['early', 'late'])
plt.yticks([0.5, 1.5], ['early', 'late'])
plt.savefig('Figure/KIRCXAIGene_LR_Confusion_matrix.png', dpi=300)
plt.close()
# Plotting the ROC curve
y_pred_proba = logreg.predict_proba(X_test)[::, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label="auc="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange', label='Baseline')
plt.legend(loc=4)
plt.title("ROC Curve")
plt.savefig('Figure/KIRCXAIGene_LR_ROC_curve.png', dpi=300)
plt.close()
# Plotting the precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba, pos_label=1)
plt.figure(figsize=(5, 5))
plt.plot(recall, precision, label="auc="+str(auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('Figure/KIRCXAIGene_LR_precision_recall_curve.png', dpi=300)
plt.close()
```
```
Accuracy score:  0.839622641509434
Recall score:  0.8295774647887324
Precision score:  0.8169504643962848
F1 score:  0.8224105646989257
Matthews correlation coefficient: 0.6464046117389293
```

## KIRC - SVC
```python
KIRC_ELgene = pd.read_csv('Data/KIRCearlylatelabelCoxProggene005.csv', index_col=0)
KIRC_Eexpr = pd.read_csv('Data/KIRCearlystageExprandClin.csv', index_col=0)
KIRC_Lexpr = pd.read_csv('Data/KIRClatestageExprandClin.csv', index_col=0)
GENEcompare = pd.read_csv('Data/probeMap_gencode.v23.annotation.gene.probemap', sep='\t')
KIRCXAI5ModelSHAPRank = pd.read_csv('Data/KIRCXAI5ModelSHAPRank.csv', index_col=0)
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
KIRC_GeneExpr = KIRC_Expr[KIRCXAI5ModelSHAPRank['Gene ensemble id']]
# Change colname(ensemble id) to gene name
KIRC_GeneExpr.columns = GENEcompare[GENEcompare['id'].isin(KIRC_GeneExpr.columns)]['gene'].values
KIRC_Target = KIRC_Expr['E_L_Stage']
# Log2 transformation
KIRC_GeneExpr = np.log2(KIRC_GeneExpr+1) # !!!!!
# Standardization 
scaler = StandardScaler() # !!!!!
KIRC_GeneExpr = scaler.fit_transform(KIRC_GeneExpr)
# Split train and test data
le = LabelEncoder()
y = le.fit_transform(KIRC_Target)
X_train, X_test, y_train, y_test = train_test_split(KIRC_GeneExpr, KIRC_Target, test_size=0.2, random_state=44)
w = compute_sample_weight(class_weight='balanced', y=y_train)
# SVC Training
svc = SVC(kernel='linear', random_state=44,probability=True)
skf = StratifiedKFold(n_splits=5, random_state=37, shuffle=True)
scores = cross_val_score(svc, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
# Fit the model
svc.fit(X_train, y_train,sample_weight=w)
y_pred = svc.predict(X_test)
# Model evaluation
start = time.time()
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
end = time.time()
```
```
Accuracy score:  0.8301886792452831
Recall score:  0.8225352112676056
Precision score:  0.8067355530042097
F1 score:  0.8131609870740306
Matthews correlation coefficient: 0.6290723849971437
```

## KIRP - Logistic regression
```python
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
GENEcompare = pd.read_csv('Data/probeMap_gencode.v23.annotation.gene.probemap', sep='\t')
KIRPXAI5ModelSHAPRank = pd.read_csv('Data/KIRPXAI5ModelSHAPRank.csv', index_col=0)
# Using regular expression to extract gene ensemble id with out version .1.2
GENEcompare['id'] = GENEcompare['id'].str.split('.').str[0]
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# check the number of early and late stage
KIRP_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRP_GeneExpr = KIRP_Expr[KIRPXAI5ModelSHAPRank['Gene ensemble id']]
# Change colname(ensemble id) to gene name
KIRP_GeneExpr.columns = GENEcompare[GENEcompare['id'].isin(KIRP_GeneExpr.columns)]['gene'].values
KIRP_Target = KIRP_Expr['E_L_Stage']
# Log2 transformation
KIRP_GeneExpr = np.log2(KIRP_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRP_GeneExpr = scaler.fit_transform(KIRP_GeneExpr)
# Split data into training and testing sets
le = LabelEncoder()
y = le.fit_transform(KIRP_Target)
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExpr,y,test_size=0.3, stratify=y,random_state=3737) #3737
# SMOTE
sm = SMOTE(random_state=44, sampling_strategy='minority',k_neighbors=40)
X_train, y_train = sm.fit_resample(X_train, y_train)
# Create a model with default parameters
logreg = LogisticRegression(random_state=44, max_iter=1000,C = 0.8, penalty = 'l2')
skf = StratifiedKFold(n_splits=10, random_state=37, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(logreg, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
# print the cross-validation scores as well as the mean of the scores and plus 95% confidence interval
mean_accuracy = np.mean(scores) * 100 
sem = stats.sem(scores) * 100  
confidence_interval = 1.96 * sem
print(f"The model achieved a 10-fold CV accuracy of {mean_accuracy:.2f}% ± {confidence_interval:.2f}%")
# Model evaluation
start = time.time()
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
plt.title("Confusion Matrix of KIRP Logistic Regression")
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks([0.5, 1.5], ['early', 'late'])
plt.yticks([0.5, 1.5], ['early', 'late'])
plt.savefig('Figure/KIRPXAIGene_LR_Confusion_matrix.png', dpi=300)
plt.close()
# Plotting the ROC curve
y_pred_proba = logreg.predict_proba(X_test)[::, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
auc = roc_auc_score(y_test, y_pred_proba)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label="auc="+str(auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='orange', label='Baseline')
plt.legend(loc=4)
plt.title("ROC Curve")
plt.savefig('Figure/KIRPXAIGene_LR_ROC_curve.png', dpi=300)
plt.close()
# Plotting the precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba, pos_label=1)
plt.figure(figsize=(5, 5))
plt.plot(recall, precision, label="auc="+str(auc))
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('Figure/KIRPXAIGene_LR_precision_recall_curve.png', dpi=300)
plt.close()
```

```
Accuracy score:  0.8255813953488372 
Recall score:  0.809134906231095
Precision score:  0.8041666666666667
F1 score:  0.8065096745162742
Matthews correlation coefficient: 0.6132814491850368
```


## KIRP - SVC
```python
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
GENEcompare = pd.read_csv('Data/probeMap_gencode.v23.annotation.gene.probemap', sep='\t')
KIRPXAI5ModelSHAPRank = pd.read_csv('Data/KIRPXAI5ModelSHAPRank.csv', index_col=0)
# Using regular expression to extract gene ensemble id with out version .1.2
GENEcompare['id'] = GENEcompare['id'].str.split('.').str[0]
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# check the number of early and late stage
KIRP_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRP_GeneExpr = KIRP_Expr[KIRPXAI5ModelSHAPRank['Gene ensemble id']]
# Change colname(ensemble id) to gene name
KIRP_GeneExpr.columns = GENEcompare[GENEcompare['id'].isin(KIRP_GeneExpr.columns)]['gene'].values
KIRP_Target = KIRP_Expr['E_L_Stage']
# Log2 transformation
KIRP_GeneExpr = np.log2(KIRP_GeneExpr+1)
# Standardization
scaler = StandardScaler()
KIRP_GeneExpr = scaler.fit_transform(KIRP_GeneExpr)
# Split data into training and testing sets
le = LabelEncoder()
y = le.fit_transform(KIRP_Target)
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExpr,y,test_size=0.3, stratify=y,random_state=888)
# SMOTE
sm = SMOTE(random_state=44, sampling_strategy='minority',k_neighbors=40)
X_train, y_train = sm.fit_resample(X_train, y_train)
# Create a model with default parameters
svc = SVC(kernel='linear', random_state=44,probability=True)
skf = StratifiedKFold(n_splits=10, random_state=37, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(svc, X_train, y_train, cv=skf, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
# Model evaluation
start = time.time()
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
end = time.time()
```
```
Accuracy score:  0.7906976744186046
 Recall score:  0.7574107683000605
 Precision score:  0.7671060891399875
 F1 score:  0.7616995073891626
 Matthews correlation coefficient: 0.5244272442318328
```