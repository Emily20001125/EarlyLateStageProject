# ExplainableAI Machine Learning

Author: Huang,Shu-Jing
Date: 2023-11-28

## environment
```shell
conda activate SHAP
cd /home/emily2835/EarlyLateStageProject/ExplainableAI_MachineLearning
```

## import package
```python
import pandas as pd
import numpy as np
import seaborn as sns
import shap
import sklearn
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.utils import compute_sample_weight
import time
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
# add column of gene name (ensemble id) to GENEcompare
GENEcompare['Gene(id)'] = GENEcompare['gene'] + '(' + GENEcompare['id'] + ')'
# add label for early and late stage
KIRC_Eexpr['E_L_Stage'] = 'early'
KIRC_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRC_Expr = pd.concat([KIRC_Eexpr,KIRC_Lexpr],axis=0)
# check the number of early and late stage
KIRC_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage 
KIRC_GeneExpr = KIRC_Expr[KIRC_ELgene[KIRC_ELgene['ProgStage']=='earlylate']['gene']]
# Change colname(ensemble id) to genename(ensemble id)
KIRC_GeneExpr.columns = GENEcompare.set_index('id').loc[KIRC_GeneExpr.columns]['Gene(id)']
KIRC_Target = KIRC_Expr['E_L_Stage']
# Log2 transformation
KIRC_GeneExprpre = np.log2(KIRC_GeneExpr+1) # !!!!!
# Standardization 
scaler = StandardScaler() # !!!!!
KIRC_GeneExprpre = scaler.fit_transform(KIRC_GeneExprpre)
# Split data into training and testing sets
y = KIRC_Target
X_train, X_test, y_train, y_test = train_test_split(KIRC_GeneExprpre,KIRC_Target,test_size=0.3, stratify=y,random_state=44) # !!!!!
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
## Explainable AI with SHAP for KIRC
```python
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(KIRC_GeneExprpre)
KIRC_feature_names = KIRC_GeneExpr.columns
# summary plot for all features
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRC_GeneExprpre, feature_names=KIRC_feature_names, plot_type='bar',max_display=20)
plt.savefig('Figure/KIRC_SHAP.png', dpi=300, bbox_inches='tight')
plt.close()
# summary plot for alive
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[0], KIRC_GeneExprpre, feature_names=KIRC_feature_names,max_display=20)
plt.savefig('Figure/KIRC_SHAP_early.png', dpi=300, bbox_inches='tight')
plt.close()
# summary plot for dead
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[1], KIRC_GeneExprpre, feature_names=KIRC_feature_names,max_display=20)
plt.savefig('Figure/KIRC_SHAP_late.png', dpi=300, bbox_inches='tight')
plt.close()
```







