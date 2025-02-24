import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

# Basic read
df = pd.read_csv('./data/equity-post-HCT-survival-predictions/train.csv')

def prepare_data(df, categorical_cols):
    # Create a copy of the dataframe
    data = df.copy()
    
    # Convert efs to integer
    data['efs'] = data['efs'].apply(int)
    
    # # Handle categorical variables by converting to 'category' dtype
    for col in categorical_cols:
        if col in data.columns:
            # Convert to string first (handles mixed types), then to category
            data[col] = data[col].astype(str).replace('nan', 'Missing').astype('category')
    
    # # Handle numerical columns (fill NaN with median)
    numerical_cols = [col for col in data.columns if col not in categorical_cols]
    for col in numerical_cols:
        if col in data.columns:
            data[col] = data[col].fillna(data[col].median())
    
    return data

def split_train_and_evaluate_with_cindex(df, train_size=0.7, val_size=0.15, test_size=0.15,
                                         categorical_cols=[
        'dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'tbi_status',
        'arrhythmia', 'graft_type', 'vent_hist', 'renal_issue', 'pulm_severe',
        'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab',
        'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity',
        'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe',
        'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match',
        'race_group', 'hepatic_mild', 'tce_div_match', 'donor_related',
        'melphalan_dose', 'cardiac', 'pulm_moderate'
    ]):
    assert train_size + val_size + test_size == 1.0, "Split sizes must sum to 1"
    
    # Prepare the data
    print(df['efs'].dtype)  # Should show int64 after apply(int)
    data = prepare_data(df, categorical_cols)
    print(data['efs'].dtype)  # Should show int64
    
    X = data.drop(['efs', 'efs_time'], axis=1)
    y = data['efs']  # Event indicator (0 for Censoring, 1 for Event)
    event_times = data['efs_time']  # Time to event
    
    # First split: Train + (Val + Test)
    X_train, X_temp, y_train, y_temp, t_train, t_temp = train_test_split(
        X, y, event_times,
        test_size=(val_size + test_size),
        random_state=42
    )
    
    # Second split: Validation and Test
    val_proportion = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test, t_val, t_test = train_test_split(
        X_temp, y_temp, t_temp,
        test_size=(1 - val_proportion),
        random_state=42
    )
    
    # Print sizes
    print(f"Training set size: {len(X_train)} ({len(X_train)/len(X):.2%})")
    print(f"Validation set size: {len(X_val)} ({len(X_val)/len(X):.2%})")
    print(f"Test set size: {len(X_test)} ({len(X_test)/len(X):.2%})")
    
    # Create DMatrix objects with enable_categorical=True
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dval = xgb.DMatrix(X_val, label=y_val, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
    
    # Define parameters
    params = {
        'objective': 'rank:pairwise',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.06,
        'n_estimators': 150,
        'min_child_weight': 4,
        'gamma': 0.1,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'random_state': 42
    }
    
    # Train the model with early stopping
    evals = [(dtrain, 'train'), (dval, 'val')]
    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=params['n_estimators'],
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False,
        # enable_categorical=True
    )
    
    # Predict probabilities for evaluation
    train_pred_proba = xgb_model.predict(dtrain)
    val_pred_proba = xgb_model.predict(dval)
    test_pred_proba = xgb_model.predict(dtest)
    
    # Calculate accuracy (threshold at 0.5)
    train_score = ((train_pred_proba > 0.5).astype(int) == y_train).mean()
    val_score = ((val_pred_proba > 0.5).astype(int) == y_val).mean()
    test_score = ((test_pred_proba > 0.5).astype(int) == y_test).mean()
    print(f"\nTraining Accuracy: {train_score:.4f}")
    print(f"Validation Accuracy: {val_score:.4f}")
    print(f"Test Accuracy: {test_score:.4f}")
    
    # Calculate C-index for the test set
    c_index = concordance_index(event_times=t_test, 
                               predicted_scores=test_pred_proba, 
                               event_observed=y_test)
    print(f"\nTest Set C-index: {c_index:.4f}")
    
    # Calculate ROC-AUC
    roc_auc = roc_auc_score(y_test, test_pred_proba)
    print(f"Test Set ROC-AUC: {roc_auc:.4f}")
    
    # Feature importance
    importance_dict = xgb_model.get_score(importance_type='weight')
    feature_importance = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    }).sort_values('importance', ascending=False)

    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    return xgb_model, X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test

# Load and evaluate the model
df = pd.read_csv('./data/equity-post-HCT-survival-predictions/train.csv')

# Call the function
model, X_train, X_val, X_test, y_train, y_val, y_test, t_train, t_val, t_test = split_train_and_evaluate_with_cindex(
    df,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15
)