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
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(KIRP_GeneExprpre,y,test_size=0.3, stratify=y,random_state=44) # !!!!!
# SMOTE
sm = SMOTE(random_state=44, sampling_strategy='minority',k_neighbors=40)  # !!!!!
X_train, y_train = sm.fit_resample(X_train, y_train)
# Best parameters
params = {'subsample': 0.7000000000000001, 'n_estimators': 500, 'min_child_weight': 1, 'max_depth': 2, 'learning_rate': 0.05, 'colsample_bytree': 0.6}
# Create a model with default parameters
xgb = xg.XGBClassifier(**params,user_label_encoder=False, eval_metric='logloss')
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=10, random_state=44, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(xgb, X_train, y_train, cv=skf, scoring='accuracy')
# print the cross-validation scores as well as the mean of the scores and plus 95% confidence interval
mean_accuracy = np.mean(scores) * 100 
sem = stats.sem(scores) * 100  
confidence_interval = 1.96 * sem
print(f"The model achieved a 10-fold CV accuracy of {mean_accuracy:.2f}% ± {confidence_interval:.2f}%")
# Model training
xgb.fit(X_train, y_train)
# Predict on test set
y_pred = xgb.predict(X_test)
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
print('Cross-validation scores:{}'.format(scores))  
# Cross-validation scores:[0.85185185 0.88888889 0.88888889 0.85185185 0.80769231 0.96153846 0.96153846 0.88461538 0.84615385 0.80769231]
print("--- %s seconds ---" % (end - start))