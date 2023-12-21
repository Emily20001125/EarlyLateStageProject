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
from imblearn.over_sampling import SMOTE
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

# Explainable AI summary with SHAP for KIRC and KIRP
## [KIRC] 
### [Ligthgbm-KIRC-Best-model] The best model for KIRC
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

[SHAP]Explainable AI with SHAP for KIRC
```python
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(KIRC_GeneExprpre)
KIRC_feature_names = KIRC_GeneExpr.columns
# save shap value for all features
save_shap = shap_values[0]
df = pd.DataFrame(save_shap, columns=KIRC_feature_names)
df.index = KIRC_Expr['sample']
df.to_csv('Output/KIRCSHAPValue.csv')
# save mean shap value for all features
# calculate the mean of all features
mean_shap_values = np.abs(save_shap).mean(axis=0)
# create a dataframe to store the mean of all features
df = pd.DataFrame(list(zip(KIRC_feature_names, mean_shap_values)), columns=['Feature', 'SHAP Value'])
# Rank the features by SHAP value
df["Gene symbol"] = df["Feature"].str.split("(").str[0]
df["Gene ensemble id"] = df["Feature"].str.split("(").str[1].str.split(")").str[0]
df = df.sort_values('SHAP Value', ascending=False)
# Exchange the order of the columns
df = df[['Feature','Gene symbol', 'Gene ensemble id','SHAP Value']]
# Output the features
df.to_csv('Output/KIRCSHAPMeanValue.csv', index=False)
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

# Force_plot for alive(plot for the j-th sample, plot for the j-th sample)
def shap_plot(i,j):
    explainerModel = shap.TreeExplainer(gbm)
    shap_values = explainerModel.shap_values(KIRC_GeneExprpre)
    plt.figure(figsize=(10, 8))
    sample = pd.DataFrame([KIRC_GeneExprpre[j,:]], columns=KIRC_feature_names)
    # Generate the force plot
    p = shap.force_plot(explainerModel.expected_value[i], shap_values[i][j,:], sample, matplotlib=True)
    # Save the plot as a file
    if i == 0:
        plt.savefig('Figure/'+str(j)+'_EarlyKIRC_SHAP.svg', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('Figure/'+str(j)+'_LateKIRC_SHAP.svg', dpi=300, bbox_inches='tight')
    plt.close()

# Call the function
shap_plot(0,2)
shap_plot(1,2)
shap_plot(0,1)

# Dependence plot for dead
def plot_dependence(model, cancertype, gene, GeneExpr, GeneExprpre):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(GeneExprpre)
    gene_column = GeneExpr.columns[GeneExpr.columns.str.contains(gene)]
    gene_column = gene_column.tolist()[0]
    gene_num = GeneExpr.columns.get_loc(gene_column)
    plt.figure(figsize=(10, 8))
    shap.dependence_plot(gene_num, shap_values[1], GeneExprpre, interaction_index=None)
    plt.xlabel(gene_column, fontsize=10)
    plt.ylabel(f'SHAP value of {gene}', fontsize=10)
    plt.savefig(f'Figure/{cancertype}SHAPlatedependenceplot{gene}.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_dependence(gbm, 'KIRC', 'FBXO48', KIRC_GeneExpr, KIRC_GeneExprpre)
plot_dependence(gbm, 'KIRC', 'GPR108', KIRC_GeneExpr, KIRC_GeneExprpre)
plot_dependence(gbm, 'KIRC', 'VCPIP1', KIRC_GeneExpr, KIRC_GeneExprpre)
```


## [KIRP]
### [Ligthgbm-KIRP-Best-model] The best model for KIRP
```python
# Read data
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
GENEcompare = pd.read_csv('Data/probeMap_gencode.v23.annotation.gene.probemap', sep='\t')
# Using regular expression to extract gene ensemble id with out version .1.2
GENEcompare['id'] = GENEcompare['id'].str.split('.').str[0]
# add column of gene name (ensemble id) to GENEcompare
GENEcompare['Gene(id)'] = GENEcompare['gene'] + '(' + GENEcompare['id'] + ')'
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# check the number of early and late stage
KIRP_Expr['E_L_Stage'].value_counts()
# Extract gene expression use regular expression for earlylate prog gene and colnames E_L_Stage
KIRP_GeneExpr = KIRP_Expr[KIRP_ELgene[KIRP_ELgene['ProgStage']=='earlylate']['gene']]
# Change colname(ensemble id) to genename(ensemble id)
KIRP_GeneExpr.columns = GENEcompare.set_index('id').loc[KIRP_GeneExpr.columns]['Gene(id)']
KIRP_Target = KIRP_Expr['E_L_Stage']
# Log2 transformation
KIRP_GeneExprpre = np.log2(KIRP_GeneExpr+1) # !!!!!
# Standardization
scaler = StandardScaler() # !!!!!
KIRP_GeneExprpre = scaler.fit_transform(KIRP_GeneExprpre)
# Split data into training and testing sets
y = KIRP_Target
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExprpre,KIRP_Target,test_size=0.3, stratify=y,random_state=44) # !!!!!
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
# Train model
gbm.fit(X_train, y_train)
# Predict on test set
y_pred = gbm.predict(X_test)
#y_pred = random_search.best_estimator_.predict(X_test)
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
```

[SHAP]Explainable AI with SHAP for KIRP
```python
explainer = shap.TreeExplainer(gbm)
shap_values = explainer.shap_values(KIRP_GeneExprpre)
KIRP_feature_names = KIRP_GeneExpr.columns
# save shap value for all features
save_shap = shap_values[0]
df = pd.DataFrame(save_shap, columns=KIRP_feature_names)
df.index = KIRP_Expr['sample']
df.to_csv('Output/KIRPSHAPValue.csv')
# save mean shap value for all features
# calculate the mean of all features
mean_shap_values = np.abs(save_shap).mean(axis=0)
# create a dataframe to store the mean of all features
df = pd.DataFrame(list(zip(KIRP_feature_names, mean_shap_values)), columns=['Feature', 'SHAP Value'])
df["Gene symbol"] = df["Feature"].str.split("(").str[0]
df["Gene ensemble id"] = df["Feature"].str.split("(").str[1].str.split(")").str[0]
# Rank the features by SHAP value
df = df.sort_values('SHAP Value', ascending=False)
# Exchange the order of the columns
df = df[['Feature','Gene symbol', 'Gene ensemble id','SHAP Value']]
# Output the features
df.to_csv('Output/KIRPSHAPMeanValue.csv', index=False)
# summary plot for all features
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, KIRP_GeneExprpre, feature_names=KIRP_feature_names, plot_type='bar',max_display=20)
plt.savefig('Figure/KIRP_SHAP.png', dpi=300, bbox_inches='tight')
plt.close()
# summary plot for alive
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[0], KIRP_GeneExprpre, feature_names=KIRP_feature_names,max_display=20)
plt.savefig('Figure/KIRP_SHAP_early.png', dpi=300, bbox_inches='tight')
plt.close()
# summary plot for dead
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values[1], KIRP_GeneExprpre, feature_names=KIRP_feature_names,max_display=20)
plt.savefig('Figure/KIRP_SHAP_late.png', dpi=300, bbox_inches='tight')
plt.close()
# Force_plot for alive(plot for the j-th sample, plot for the j-th sample)
def shap_plot(i,j):
    explainerModel = shap.TreeExplainer(gbm)
    shap_values = explainerModel.shap_values(KIRP_GeneExprpre)
    plt.figure(figsize=(10, 8))
    sample = pd.DataFrame([KIRP_GeneExprpre[j,:]], columns=KIRP_feature_names)
    # Generate the force plot
    p = shap.force_plot(explainerModel.expected_value[i], shap_values[i][j,:], sample, matplotlib=True)
    # Save the plot as a file
    if i == 0:
        plt.savefig('Figure/'+str(j)+'_EarlyKIRP_SHAP.svg', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('Figure/'+str(j)+'_LateKIRP_SHAP.svg', dpi=300, bbox_inches='tight')
    plt.close()

# Call the function
shap_plot(0,2)
shap_plot(1,2)
shap_plot(0,88)
shap_plot(1,88)

# Dependence plot for dead
def plot_dependence(model, cancertype, gene, GeneExpr, GeneExprpre):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(GeneExprpre)
    gene_column = GeneExpr.columns[GeneExpr.columns.str.contains(gene)]
    gene_column = gene_column.tolist()[0]
    gene_num = GeneExpr.columns.get_loc(gene_column)
    plt.figure(figsize=(10, 8))
    shap.dependence_plot(gene_num, shap_values[1], GeneExprpre, interaction_index=None)
    plt.xlabel(gene_column, fontsize=10)
    plt.ylabel(f'SHAP value of {gene}', fontsize=10)
    plt.savefig(f'Figure/{cancertype}SHAPlatedependenceplot{gene}.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_dependence(gbm, 'KIRP', 'NEIL3', KIRP_GeneExpr, KIRP_GeneExprpre)
plot_dependence(gbm, 'KIRP', 'FXR1', KIRP_GeneExpr, KIRP_GeneExprpre)
plot_dependence(gbm, 'KIRP', 'C1orf112', KIRP_GeneExpr, KIRP_GeneExprpre)
```


# [Check] if there is duplicated patient between early and late stage
Check if there is duplicated patient between early and late stage in KIRC
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
# Extract X_PATIENT(ex:TCGA-2K-A9WE) last 4 character as patient id
KIRC_Expr['X_PATIENT_sign'] = KIRC_Expr['X_PATIENT'].str[-4:]
#Cheack if there is duplicated patient between early and late stage
KIRC_Expr.duplicated(subset=['X_PATIENT_sign'], keep=False).value_counts()
KIRC_Expr['X_PATIENT_sign'].nunique()
```
```
!!! No duplicated patient in KIRC
```
Check if there is duplicated patient between early and late stage in KIRP
```python
# Read data
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
GENEcompare = pd.read_csv('Data/probeMap_gencode.v23.annotation.gene.probemap', sep='\t')
# Using regular expression to extract gene ensemble id with out version .1.2
GENEcompare['id'] = GENEcompare['id'].str.split('.').str[0]
# add column of gene name (ensemble id) to GENEcompare
GENEcompare['Gene(id)'] = GENEcompare['gene'] + '(' + GENEcompare['id'] + ')'
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# Extract X_PATIENT(ex:TCGA-2K-A9WE) last 4 character as patient id
KIRP_Expr['X_PATIENT_sign'] = KIRP_Expr['X_PATIENT'].str[-4:]
#Cheack if there is duplicated patient between early and late stage
KIRP_Expr.duplicated(subset=['X_PATIENT_sign'], keep=False).value_counts()
KIRP_Expr['X_PATIENT_sign'].nunique()
```
```
!!! No duplicated patient in KIRP
```


# [Top10-genes-Heatmap] Top 10 genes heatmap for KIRC and KIRP
## environment
```shell
conda activate python3_10
cd /home/emily2835/EarlyLateStageProject/ExplainableAI_MachineLearning
```
## 1. Data preparation
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot as plt
```
## [KIRP] Top 10 genes heatmap for KIRP
```python
# Extract top 10 genes expression
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
GENEcompare = pd.read_csv('Data/probeMap_gencode.v23.annotation.gene.probemap', sep='\t')
# Using regular expression to extract gene ensemble id with out version .1.2
GENEcompare['id'] = GENEcompare['id'].str.split('.').str[0]
# add column of gene name (ensemble id) to GENEcompare
GENEcompare['Gene(id)'] = GENEcompare['gene'] + '(' + GENEcompare['id'] + ')'
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# Import data
KIRPSHAPMeanValue = pd.read_csv('Output/KIRPSHAPMeanValue.csv')
# Extract top 10 genes
KIRPSHAPMeanValue_top10 = KIRPSHAPMeanValue.head(10)
# Extract top 10 genes expression (KIRPSHAPMeanValue_top10)
KIRP_GeneExpr = KIRP_Expr[KIRPSHAPMeanValue_top10["Gene ensemble id"]]
# Change colname(ensemble id) to genename
KIRP_GeneExpr.columns = KIRPSHAPMeanValue_top10["Gene symbol"]
# Log2 transformation
KIRP_GeneExprpre = np.log2(KIRP_GeneExpr+1)
#  Normalize it by row:
KIRP_GeneExprpre = KIRP_GeneExprpre.apply(lambda x: (x-x.mean())/x.std(), axis = 0)
# add stage label and cancer type label
KIRP_GeneExprpre.index = KIRP_Expr['E_L_Stage']
# Plotting Heatmap
plt.figure(figsize=(40,30))
sns.set(font_scale=0.5)
sns.heatmap(KIRP_GeneExprpre, cmap="seismic",square=True)
plt.xlabel('Gene symbol', fontsize=40)
plt.ylabel('Stage', fontsize=40)
plt.savefig('Figure/KIRPTop10genesHeatmap.png', dpi=300)
plt.savefig('Figure/KIRPTop10genesHeatmap.svg', dpi=300)
plt.close()
```


## [KIRC] Top 10 genes heatmap for KIRC
```python
# Extract top 10 genes expression
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
# Import data
KIRCSHAPMeanValue = pd.read_csv('Output/KIRCSHAPMeanValue.csv')
# Extract top 10 genes
KIRCSHAPMeanValue_top10 = KIRCSHAPMeanValue.head(10)
# Extract top 10 genes expression (KIRCSHAPMeanValue_top10)
KIRC_GeneExpr = KIRC_Expr[KIRCSHAPMeanValue_top10["Gene ensemble id"]]
# Change colname(ensemble id) to genename
KIRC_GeneExpr.columns = KIRCSHAPMeanValue_top10["Gene symbol"]
# Log2 transformation
KIRC_GeneExprpre = np.log2(KIRC_GeneExpr+1)
#  Normalize it by row:
KIRC_GeneExprpre = KIRC_GeneExprpre.apply(lambda x: (x-x.mean())/x.std(), axis = 0)
# add stage label and cancer type label
KIRC_GeneExprpre.index = KIRC_Expr['E_L_Stage']
# Plotting Heatmap
plt.figure(figsize=(40,30))
sns.set(font_scale=0.5)
sns.heatmap(KIRC_GeneExprpre, cmap="seismic",square=True)
plt.xlabel('Gene symbol', fontsize=40)
plt.ylabel('Stage', fontsize=40)
plt.savefig('Figure/KIRCTop10genesHeatmap.png', dpi=300)
plt.savefig('Figure/KIRCTop10genesHeatmap.svg', dpi=300)
plt.close()
```

 
## [KIRP] Top 10 genes boxplot for KIRP (each gene)
```python
# Extract top 10 genes expression
# Extract top 10 genes expression
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
GENEcompare = pd.read_csv('Data/probeMap_gencode.v23.annotation.gene.probemap', sep='\t')
# Using regular expression to extract gene ensemble id with out version .1.2
GENEcompare['id'] = GENEcompare['id'].str.split('.').str[0]
# add column of gene name (ensemble id) to GENEcompare
GENEcompare['Gene(id)'] = GENEcompare['gene'] + '(' + GENEcompare['id'] + ')'
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# Import data
KIRPSHAPMeanValue = pd.read_csv('Output/KIRPSHAPMeanValue.csv')
# Extract top 10 genes
KIRPSHAPMeanValue_top10 = KIRPSHAPMeanValue.head(10)
# Extract top 10 genes expression (KIRPSHAPMeanValue_top10)
KIRP_GeneExpr = KIRP_Expr[KIRPSHAPMeanValue_top10["Gene ensemble id"]]
# Change colname(ensemble id) to genename
KIRP_GeneExpr.columns = KIRPSHAPMeanValue_top10["Gene symbol"]
# Log2 transformation
KIRP_GeneExprpre = np.log2(KIRP_GeneExpr+1)
# add stage label and cancer type label
KIRP_GeneExprpre['E_L_Stage'] = KIRP_Expr['E_L_Stage']
# Transform
df_long = pd.melt(KIRP_GeneExprpre, id_vars='E_L_Stage', var_name='Gene_symbol', value_name='Value')
# plot the boxplot for each gene
plt.figure(figsize=(5,6))
sns.set(font_scale=1, style="white")
plotdata = df_long[df_long['Gene_symbol']=="NEIL3"]
sns.boxplot(x="Gene_symbol", y="Value", hue="E_L_Stage", data=plotdata, palette=sns.color_palette(), width=0.4)
plt.savefig('Figure/KIRPTop10genesBoxplot.png', dpi=300)
plt.close()
```

## [KIRC] Top 5% genes heatmap for KIRC
```python
# Extract top 5% genes expression
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# Import data
KIRPSHAPMeanValue = pd.read_csv('Output/KIRPSHAPMeanValue.csv')
# Extract top 5% genes
KIRPSHAPMeanValue_5p = KIRPSHAPMeanValue.head(int(len(KIRPSHAPMeanValue)*0.05))
# Extract top 10 genes expression (KIRPSHAPMeanValue_top10)
KIRP_GeneExpr = KIRP_Expr[KIRPSHAPMeanValue_5p["Gene ensemble id"]]
# Change colname(ensemble id) to genename
KIRP_GeneExpr.columns = KIRPSHAPMeanValue_5p["Gene symbol"]
# Log2 transformation
KIRP_GeneExprpre = np.log2(KIRP_GeneExpr+1)
#  Normalize it by row:
#KIRP_GeneExprpre = KIRP_GeneExprpre.apply(lambda x: (x-x.mean())/x.std(), axis = 0)
# add stage label and cancer type label
KIRP_GeneExprpre['E_L_Stage'] = KIRP_Expr['E_L_Stage']
# Set row colors by E_L_Stage
my_palette = dict(zip(KIRP_GeneExprpre.E_L_Stage.unique(), ["red","blue"]))
row_colors = KIRP_GeneExprpre['E_L_Stage'].map(my_palette)
# extract top 5% genes expression without E_L_Stage
KIRP_GeneExprpre = KIRP_GeneExprpre.drop(['E_L_Stage'], axis=1)
# Plotting Heatmap
plt.figure(figsize=(40,40))
sns.clustermap(KIRP_GeneExprpre,cmap="seismic",row_cluster=False, col_cluster=False,z_score=1,row_colors=row_colors ,yticklabels=False,xticklabels=False)
plt.savefig('Figure/KIRPTop5pGenesHeatmap.png', dpi=300)
plt.savefig('Figure/KIRPTop5pGenesHeatmap.svg', dpi=300)
plt.close()
```

## [KIRC] Last 5% genes heatmap for KIRP
```python
# Extract last 5% genes expression
KIRP_ELgene = pd.read_csv('Data/KIRPearlylatelabelCoxProggene005.csv', index_col=0)
KIRP_Eexpr = pd.read_csv('Data/KIRPearlystageExprandClin.csv', index_col=0)
KIRP_Lexpr = pd.read_csv('Data/KIRPlatestageExprandClin.csv', index_col=0)
# add label for early and late stage
KIRP_Eexpr['E_L_Stage'] = 'early'
KIRP_Lexpr['E_L_Stage'] = 'late'
# Combine early and late stage
KIRP_Expr = pd.concat([KIRP_Eexpr,KIRP_Lexpr],axis=0)
# Import data
KIRPSHAPMeanValue = pd.read_csv('Output/KIRPSHAPMeanValue.csv')
# Extract last 5% genes
KIRPSHAPMeanValue_5p = KIRPSHAPMeanValue.tail(int(len(KIRPSHAPMeanValue)*0.05))
# Extract last 5% genes expression (KIRPSHAPMeanValue_5p)
KIRP_GeneExpr = KIRP_Expr[KIRPSHAPMeanValue_5p["Gene ensemble id"]]
# Change colname(ensemble id) to genename
KIRP_GeneExpr.columns = KIRPSHAPMeanValue_5p["Gene symbol"]
# Log2 transformation
KIRP_GeneExprpre = np.log2(KIRP_GeneExpr+1)
# add stage label and cancer type label
KIRP_GeneExprpre['E_L_Stage'] = KIRP_Expr['E_L_Stage']
# Set row colors by E_L_Stage
my_palette = dict(zip(KIRP_GeneExprpre.E_L_Stage.unique(), ["red","blue"]))
row_colors = KIRP_GeneExprpre['E_L_Stage'].map(my_palette)
# extract last 5% genes expression without E_L_Stage
KIRP_GeneExprpre = KIRP_GeneExprpre.drop(['E_L_Stage'], axis=1)
# Plotting Heatmap
plt.figure(figsize=(40,30))
sns.set(font_scale=1)
sns.clustermap(KIRP_GeneExprpre, cmap="seismic",row_cluster=False, col_cluster=False,z_score=1,row_colors=row_colors ,yticklabels=False,xticklabels=False)
plt.savefig('Figure/KIRPLast5pGenesHeatmap.png', dpi=300)
plt.savefig('Figure/KIRPLast5pGenesHeatmap.svg', dpi=300)
plt.close()
```

## Function`plot_heatmap_for_toporlast`: plot heatmap for top or last genes
plot heatmap for top genes
```python
def plot_gene_expression_top(cancer_type, percentage):
    ELgene = pd.read_csv(f'Data/{cancer_type}earlylatelabelCoxProggene005.csv', index_col=0)
    Eexpr = pd.read_csv(f'Data/{cancer_type}earlystageExprandClin.csv', index_col=0)
    Lexpr = pd.read_csv(f'Data/{cancer_type}latestageExprandClin.csv', index_col=0)
    Eexpr['E_L_Stage'] = 'early'
    Lexpr['E_L_Stage'] = 'late'
    Expr = pd.concat([Eexpr,Lexpr],axis=0)
    SHAPMeanValue = pd.read_csv(f'Output/{cancer_type}SHAPMeanValue.csv')
    SHAPMeanValue_p = SHAPMeanValue.head(int(len(SHAPMeanValue)*percentage))
    GeneExpr = Expr[SHAPMeanValue_p["Gene ensemble id"]]
    GeneExpr.columns = SHAPMeanValue_p["Gene symbol"]
    GeneExprpre = np.log2(GeneExpr+1)
    GeneExprpre['E_L_Stage'] = Expr['E_L_Stage']
    my_palette = dict(zip(GeneExprpre.E_L_Stage.unique(), ["blue","red"]))
    row_colors = GeneExprpre['E_L_Stage'].map(my_palette)
    GeneExprpre = GeneExprpre.drop(['E_L_Stage'], axis=1)
    plt.figure(figsize=(40,40))
    sns.clustermap(GeneExprpre,cmap="seismic",row_cluster=False, col_cluster=False,z_score=1,row_colors=row_colors ,yticklabels=False,xticklabels=False)
    plt.savefig(f'Figure/{cancer_type}top{percentage}pGenesHeatmap.png', dpi=300)
    plt.savefig(f'Figure/{cancer_type}top{percentage}pGenesHeatmap.svg', dpi=300)
    plt.close()

plot_gene_expression_top('KIRP', 0.05)
plot_gene_expression_top('KIRC', 0.01)
```

plot heatmap for last genes
```python
def plot_gene_expression_last(cancer_type, percentage):
    ELgene = pd.read_csv(f'Data/{cancer_type}earlylatelabelCoxProggene005.csv', index_col=0)
    Eexpr = pd.read_csv(f'Data/{cancer_type}earlystageExprandClin.csv', index_col=0)
    Lexpr = pd.read_csv(f'Data/{cancer_type}latestageExprandClin.csv', index_col=0)
    Eexpr['E_L_Stage'] = 'early'
    Lexpr['E_L_Stage'] = 'late'
    Expr = pd.concat([Eexpr,Lexpr],axis=0)
    SHAPMeanValue = pd.read_csv(f'Output/{cancer_type}SHAPMeanValue.csv')
    SHAPMeanValue_p = SHAPMeanValue.tail(int(len(SHAPMeanValue)*percentage))
    # print the genes number
    print(len(SHAPMeanValue_p))
    GeneExpr = Expr[SHAPMeanValue_p["Gene ensemble id"]]
    GeneExpr.columns = SHAPMeanValue_p["Gene symbol"]
    GeneExprpre = np.log2(GeneExpr+1)
    GeneExprpre['E_L_Stage'] = Expr['E_L_Stage']
    my_palette = dict(zip(GeneExprpre.E_L_Stage.unique(), ["blue","red"]))
    row_colors = GeneExprpre['E_L_Stage'].map(my_palette)
    GeneExprpre = GeneExprpre.drop(['E_L_Stage'], axis=1)
    plt.figure(figsize=(40,40))
    sns.clustermap(GeneExprpre,cmap="seismic",row_cluster=False, col_cluster=False,z_score=1,row_colors=row_colors ,yticklabels=False,xticklabels=False)
    plt.savefig(f'Figure/{cancer_type}last{percentage}pGenesHeatmap.png', dpi=300)
    plt.savefig(f'Figure/{cancer_type}last{percentage}pGenesHeatmap.svg', dpi=300)
    plt.close()

plot_gene_expression_last('KIRP', 0.05)
plot_gene_expression_last('KIRC', 0.01)
```

# [Scatterplot] Scatterplot for KIRC and KIRP SHAP value and gene expression difference
```python
cancer_type = 'KIRC'
ELgene = pd.read_csv(f'Data/{cancer_type}earlylatelabelCoxProggene005.csv', index_col=0)
Eexpr = pd.read_csv(f'Data/{cancer_type}earlystageExprandClin.csv', index_col=0)
Lexpr = pd.read_csv(f'Data/{cancer_type}latestageExprandClin.csv', index_col=0)
Eexpr['E_L_Stage'] = 'early'
Lexpr['E_L_Stage'] = 'late'
Expr = pd.concat([Eexpr,Lexpr],axis=0)
SHAPMeanValue = pd.read_csv(f'Output/{cancer_type}SHAPMeanValue.csv')
# percentage = 0.05
# SHAPMeanValue= SHAPMeanValue.head(int(len(SHAPMeanValue)*percentage))
GeneExpr = Expr[SHAPMeanValue["Gene ensemble id"]]
GeneExpr.columns = SHAPMeanValue["Gene symbol"]
# Log2 transformation
GeneExprpre = np.log2(GeneExpr+1)
# z score
GeneExprpre = GeneExprpre.apply(lambda x: (x-x.mean())/x.std(), axis = 1)
# add stage label and cancer type label
GeneExprpre['E_L_Stage'] = Expr['E_L_Stage']
# Calculate the mean of early and late stage
Genemean = GeneExprpre.groupby('E_L_Stage').median()
# Transpose
Genemean = Genemean.T
Genemean['diff'] = Genemean['late'] - Genemean['early']
# abs 
Genemean['diff'] = Genemean['diff'].abs()
# Combine SHAP value and gene expression difference
com = pd.merge(SHAPMeanValue, Genemean, left_on='Gene symbol', right_on='Gene symbol')
# Plotting the scatterplot
plt.figure(figsize=(10, 8))
sns.scatterplot(x='early', y='late', data=com)
plt.savefig(f'Figure/{cancer_type}SHAPdiffScatterplot.png', dpi=300)
```

