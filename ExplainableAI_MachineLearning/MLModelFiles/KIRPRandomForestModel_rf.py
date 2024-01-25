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
# print the cross-validation scores as well as the mean of the scores and plus 95% confidence interval
mean_accuracy = np.mean(scores) * 100 
sem = stats.sem(scores) * 100  
confidence_interval = 1.96 * sem
print(f"The model achieved a 10-fold CV accuracy of {mean_accuracy:.2f}% ± {confidence_interval:.2f}%")
# fit model
start = time.time()
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
# Cross-validation scores:[0.85185185 0.88888889 0.88888889 0.85185185 0.80769231 0.96153846 0.96153846 0.88461538 0.84615385 0.80769231]
print("--- %s seconds ---" % (end - start))