import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import argparse
import json


def create_submission(predictions, ids, output_path):
    """
    Create submission file in the required format
    """
    submission = pd.DataFrame({
        'ID': ids,
        'prediction': predictions
    })
    
    submission.to_csv(output_path, index=False)
    print(f"Submission file created at {output_path}")
    
    return submission

def prepare_test_data(df, categorical_cols, column_names):
    """
    Prepare test data ensuring it has all the columns the model was trained on
    
    Args:
        df: Test dataframe
        categorical_cols: List of categorical column names before encoding
        column_names: Complete list of column names after one-hot encoding that model expects
    """
    # Create a copy of the dataframe
    data = df.copy()
    
    # Store IDs for submission
    ids = data['ID'].values if 'ID' in data.columns else np.arange(len(data))
    
    # Define preprocessing for categorical and numerical columns
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_cols = [col for col in data.columns if col not in categorical_cols and col != 'ID']
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols),
            ('num', numerical_transformer, numerical_cols)
        ],
        remainder='drop'  # Drop columns like ID
    )
    
    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(data)
    
    # Get feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_cols)
    feature_names = np.concatenate([cat_feature_names, numerical_cols])
    
    # Check if we have all required columns
    missing_columns = set(column_names) - set(feature_names)
    if missing_columns:
        print(f"Warning: {len(missing_columns)} columns are missing in the test data:")
        print(list(missing_columns)[:10], "..." if len(missing_columns) > 10 else "")
        
        # Create a DataFrame with the correct columns (fill missing with zeros)
        X_final = np.zeros((X_preprocessed.shape[0], len(column_names)))
        
        # Map feature positions
        for i, feature in enumerate(feature_names):
            if feature in column_names:
                col_idx = list(column_names).index(feature)
                X_final[:, col_idx] = X_preprocessed[:, i]
    else:
        # Reorder columns to match training data
        X_final = np.zeros((X_preprocessed.shape[0], len(column_names)))
        for i, feature in enumerate(column_names):
            if feature in feature_names:
                col_idx = list(feature_names).index(feature)
                X_final[:, i] = X_preprocessed[:, col_idx]
    
    return X_final, ids

def main(test_path, model_path, output_path):
    """
    Main function to generate predictions
    
    Args:
        test_path: Path to test CSV
        model_path: Path to saved model
        output_path: Path to save predictions
    """
    print(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)
    print(f"Test data shape: {test_df.shape}")
    
    # Load the model
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Define categorical columns (same as in training)
    categorical_cols = [
        'dri_score', 'psych_disturb', 'cyto_score', 'diabetes', 'tbi_status',
        'arrhythmia', 'graft_type', 'vent_hist', 'renal_issue', 'pulm_severe',
        'prim_disease_hct', 'cmv_status', 'tce_imm_match', 'rituximab',
        'prod_type', 'cyto_score_detail', 'conditioning_intensity', 'ethnicity',
        'obesity', 'mrd_hct', 'in_vivo_tcd', 'tce_match', 'hepatic_severe',
        'prior_tumor', 'peptic_ulcer', 'gvhd_proph', 'rheum_issue', 'sex_match',
        'race_group', 'hepatic_mild', 'tce_div_match', 'donor_related',
        'melphalan_dose', 'cardiac', 'pulm_moderate'
    ]
    
    with open('columns.json', 'r') as f:
        column_names = json.load(f)
    
    print(f"Model expects {len(column_names)} columns after preprocessing")
    
    print(f"Model expects {len(column_names)} columns after preprocessing")
    
    # Prepare test data
    X_test, ids = prepare_test_data(test_df, categorical_cols, column_names)
    print(f"Preprocessed test data shape: {X_test.shape}")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = model.predict(X_test)
    
    # Create submission file
    submission = create_submission(predictions, ids, output_path)
    
    print(f"Prediction range: {predictions.min()} to {predictions.max()}")
    print(f"Total predictions: {len(predictions)}")
    
    return submission

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions for survival analysis')
    parser.add_argument('--test_path', type=str, default='./data/equity-post-HCT-survival-predictions/test.csv',
                        help='Path to test CSV file')
    parser.add_argument('--model_path', type=str, default='hyperopt_results/best_model.pkl',
                        help='Path to saved model file')
    parser.add_argument('--output_path', type=str, default='submission.csv',
                        help='Path to save prediction output')
    
    args = parser.parse_args()
    
    main(args.test_path, args.model_path, args.output_path)