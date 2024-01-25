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
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(KIRC_GeneExprpre,y,test_size=0.3, stratify=y,random_state=44)
# Create a model with default parameters
params = {'n_estimators': 710, 'min_samples_split': 9, 'min_samples_leaf': 3, 'max_samples': 0.9052631578947369, 'max_depth': 6} 
rf = RandomForestClassifier(**params,random_state=44,class_weight='balanced')
# Create a StratifiedKFold object
skf = StratifiedKFold(n_splits=10, random_state=44, shuffle=True)
# Initialize the list to store the scores
scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy')
# print the cross-validation scores as well as the mean of the scores and plus 95% confidence interval
mean_accuracy = np.mean(scores) * 100 
sem = stats.sem(scores) * 100  
confidence_interval = 1.96 * sem
print(f"The model achieved a 10-fold CV accuracy of {mean_accuracy:.2f}% ± {confidence_interval:.2f}%")
start = time.time()
# fit model
rf.fit(X_train, y_train)
# Predict on test set
y_pred = rf.predict(X_test)
end = time.time()
print('Confusion matrix: ', confusion_matrix(y_test, y_pred))
print('Accuracy score: ', accuracy_score(y_test, y_pred), '\n',
      'Recall score: ', recall_score(y_test, y_pred, average='macro'), '\n',
      'Precision score: ', precision_score(y_test, y_pred, average='macro'), '\n',
      'F1 score: ', f1_score(y_test, y_pred, average='macro'), '\n',
      'Matthews correlation coefficient:', matthews_corrcoef(y_test, y_pred))
print('Cross-validation scores:{}'.format(scores))
# Cross-validation scores:[0.72972973 0.72972973 0.83783784 0.83783784 0.75675676 0.7027027 0.72972973 0.72972973 0.81081081 0.86111111]
print("--- %s seconds ---" % (end - start))