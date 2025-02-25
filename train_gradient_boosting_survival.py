

# Load data
df = pd.read_csv('./data/equity-post-HCT-survival-predictions/train.csv')

def prepare_data(df, categorical_cols, id_col='ID'):
    # Create a copy of the dataframe
    data = df.copy()
    
    # Ensure efs is integer (event indicator: 0 or 1)
    data['efs'] = data['efs'].astype(int)
    
    # Drop the ID column if it exists
    if id_col in data.columns:
        data = data.drop(columns=[id_col])
        print(f"Dropped column: {id_col}")
    else:
        print(f"No column named '{id_col}' found in the dataset")
    
    # Separate features and target
    X = data.drop(['efs', 'efs_time'], axis=1)
    y = Surv.from_arrays(event=data['efs'], time=data['efs_time'])
    
    # Define preprocessing for categorical and numerical columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ])
    
    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Get feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([cat_feature_names, numerical_cols])
    
    return X_preprocessed, y, feature_names, preprocessor

def split_train_and_evaluate_with_sksurv(df, train_size=0.7, val_size=0.15, test_size=0.15,
                                         categorical_cols=[
        'dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'tbi_status',
        'arrhythmia', 'graft_type', 'vent_hist', 'renal_issue', 'pulm_severe',
        'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab',
        'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity',
        'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe',
        'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match',
        'race_group', 'hepatic_mild', 'tce_div_match', 'donor_related',
        'melphalan_dose', 'cardiac', 'pulm_moderate'
    ], id_col='ID'):
    assert train_size + val_size + test_size == 1.0, "Split sizes must sum to 1"
    
    # Prepare data
    X, y, feature_names, preprocessor = prepare_data(df, categorical_cols, id_col)
    
    # Split into train + (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=42
    )
    
    # Split temp into validation and test
    val_proportion = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_proportion), random_state=42
    )
    
    # Print sizes
    print(f"Training set size: {len(X_train)} ({len(X_train)/len(X):.2%})")
    print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X):.2%})")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X):.2%})")
    
    # Define and train the model
    model = GradientBoostingSurvivalAnalysis(
        n_estimators=150,
        learning_rate=0.06,
        max_depth=8,
        min_samples_split=4,
        subsample=0.7,
        random_state=42,
        max_features='log2', # 'sqrt', 0.3-0.7, 
        n_iter_no_change=10, # Set to 10, 20, or 50, and pair with a validation fraction
        validation_fraction=0.1, # 0.1–0.3
        verbose=1
    )
    
    model.fit(X_train, y_train)
    
    # Predict risk scores
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    
    # Extract event times and indicators for evaluation
    t_train = y_train['time']
    e_train = y_train['event']
    t_val = y_val['time']
    e_val = y_val['event']
    t_test = y_test['time']
    e_test = y_test['event']
    
    # Calculate C-index
    c_index_train = concordance_index(t_train, -train_pred, e_train)
    c_index_val = concordance_index(t_val, -val_pred, e_val)
    c_index_test = concordance_index(t_test, -test_pred, e_test)
    print(f"\nTraining C-index: {c_index_train:.4f}")
    print(f"Validation C-index: {c_index_val:.4f}")
    print(f"Test C-index: {c_index_test:.4f}")
    
    # Calculate ROC-AUC
    roc_auc_test = roc_auc_score(e_test, test_pred)
    print(f"Test Set ROC-AUC: {roc_auc_test:.4f}")
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 most important features:")
    print(importance.head(10))
    
    return model, preprocessor, X_train, X_val, X_test, y_train, y_val, y_test

# Load and evaluate the model
df = pd.read_csv('./data/equity-post-HCT-survival-predictions/train.csv')

# Call the function, specifying the ID column to remove
model, preprocessor, X_train, X_val, X_test, y_train, y_val, y_test = split_train_and_evaluate_with_sksurv(
    df,
    train_size=0.75,
    val_size=0.10,
    test_size=0.15,
    id_col='ID'  # Adjust this to match your actual ID column name
)