import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence, plot_objective

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

def split_data_for_hyperopt(df, train_size=0.7, val_size=0.15, test_size=0.15,
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
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

def evaluate_model(model, X, y):
    """Evaluate model using concordance index"""
    prediction = model.predict(X)
    
    # Extract event indicators and times
    event = y['event']
    time = y['time']
    
    # Calculate concordance index
    c_index, _, _, _, _ = concordance_index_censored(event, time, -prediction)
    
    return c_index

def hyperparameter_optimization(X_train, X_val, y_train, y_val, feature_names, n_calls=50):
    """
    Optimize hyperparameters using Bayesian optimization.
    
    Args:
        X_train: Training features
        X_val: Validation features
        y_train: Training target
        y_val: Validation target
        feature_names: Names of features
        n_calls: Number of optimization iterations
        
    Returns:
        dict: Optimization results
    """
    # Create output directory for results
    os.makedirs('hyperopt_results', exist_ok=True)
    
    # Define the hyperparameter search space with explicit names
    space = [
        Integer(50, 300, name='n_estimators'),
        Real(0.01, 0.2, prior='log-uniform', name='learning_rate'),
        Integer(3, 10, name='max_depth'),
        Integer(2, 10, name='min_samples_split'),
        Real(0.5, 1.0, name='subsample'),
        Categorical(['sqrt', 'log2', None], name='max_features'),
        Integer(5, 20, name='n_iter_no_change'),
        Real(0.1, 0.3, name='validation_fraction')
    ]
    
    # Parameter names for reference
    param_names = ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split', 
                   'subsample', 'max_features', 'n_iter_no_change', 'validation_fraction']
    
    # Objective function to minimize (negative c-index)
    @use_named_args(space)
    def objective(n_estimators, learning_rate, max_depth, min_samples_split, 
                  subsample, max_features, n_iter_no_change, validation_fraction):
        # Create model with current parameters
        model = GradientBoostingSurvivalAnalysis(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            subsample=subsample,
            max_features=max_features,
            n_iter_no_change=n_iter_no_change,
            validation_fraction=validation_fraction,
            random_state=42,
            verbose=0
        )
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        c_index = evaluate_model(model, X_val, y_val)
        
        # We want to maximize c-index, but skopt minimizes
        return -c_index
    
    # Progress tracking
    results = {
        'iteration': [],
        'params': [],
        'c_index': [],
        'best_c_index': []
    }
    
    # Callback function to track progress
    def on_step(res):
        # Extract current iteration results
        iteration = len(results['iteration'])
        
        # Convert X values to named parameters
        current_params = {}
        for i, name in enumerate(param_names):
            current_params[name] = res.x_iters[-1][i]
        
        current_c_index = -res.func_vals[-1]
        best_c_index = -res.fun
        
        # Store results
        results['iteration'].append(iteration)
        results['params'].append(current_params)
        results['c_index'].append(current_c_index)
        results['best_c_index'].append(best_c_index)
        
        # Print progress
        print(f"Iteration {iteration+1}/{n_calls}: C-index = {current_c_index:.4f} | Best = {best_c_index:.4f}")
        
        # Create progress plot every 5 iterations
        if (iteration + 1) % 5 == 0 or iteration == n_calls - 1:
            plt.figure(figsize=(12, 6))
            plt.plot(results['iteration'], results['c_index'], 'o-', label='Current C-index')
            plt.plot(results['iteration'], results['best_c_index'], 'o-', label='Best C-index')
            plt.xlabel('Iteration')
            plt.ylabel('C-index')
            plt.title('Hyperparameter Optimization Progress')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'hyperopt_results/optimization_progress.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    # Run Bayesian optimization
    print(f"Starting hyperparameter optimization with {n_calls} iterations...")
    start_time = time.time()
    
    optimization_result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_initial_points=10,
        callback=on_step,
        random_state=42,
        verbose=True
    )
    
    optimization_time = time.time() - start_time
    print(f"Optimization completed in {optimization_time:.2f} seconds")
    
    # Get best parameters
    best_params = {}
    for i, name in enumerate(param_names):
        best_params[name] = optimization_result.x[i]
    
    best_c_index = -optimization_result.fun
    
    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best validation C-index: {best_c_index:.4f}")
    
    # Train final model with best parameters
    best_model = GradientBoostingSurvivalAnalysis(
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        subsample=best_params['subsample'],
        max_features=best_params['max_features'],
        n_iter_no_change=best_params['n_iter_no_change'],
        validation_fraction=best_params['validation_fraction'],
        random_state=42
    )
    best_model.fit(X_train, y_train)
    
    # Save best model and parameters
    joblib.dump(best_model, 'hyperopt_results/best_model.pkl')
    joblib.dump(best_params, 'hyperopt_results/best_params.pkl')
    
    # Create visualization plots
    visualize_results(optimization_result, param_names, best_model, feature_names)
    
    return {
        'best_params': best_params,
        'best_c_index': best_c_index,
        'best_model': best_model,
        'optimization_result': optimization_result,
        'results': results
    }

def visualize_results(optimization_result, param_names, best_model, feature_names):
    """Create visualizations for optimization results"""
    # Create directory for plots
    os.makedirs('hyperopt_results/plots', exist_ok=True)
    
    # 1. Convergence plot
    plt.figure(figsize=(10, 6))
    plot_convergence(optimization_result)
    plt.title("Convergence plot")
    plt.savefig('hyperopt_results/plots/convergence.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Parameter importance and relationships
    try:
        plot_objective(optimization_result, n_points=10)
        plt.savefig('hyperopt_results/plots/parameter_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Warning: Could not create parameter importance plot: {str(e)}")
    
    # 3. Feature importance
    if hasattr(best_model, 'feature_importances_'):
        # Get top 20 features
        n_features = min(20, len(feature_names))
        indices = np.argsort(best_model.feature_importances_)[::-1][:n_features]
        
        plt.figure(figsize=(12, 8))
        plt.title('Top 20 Feature Importances')
        plt.barh(range(n_features), best_model.feature_importances_[indices])
        plt.yticks(range(n_features), [feature_names[i] for i in indices])
        plt.tight_layout()
        plt.savefig('hyperopt_results/plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Parameter distributions
    fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 2*len(param_names)))
    for i, (name, ax) in enumerate(zip(param_names, axes)):
        x = [x_iter[i] for x_iter in optimization_result.x_iters]
        y = optimization_result.func_vals
        
        # Handle categorical parameters differently
        if isinstance(optimization_result.space.dimensions[i], Categorical):
            categories = optimization_result.space.dimensions[i].categories
            # Convert categorical values to indices for plotting
            cat_indices = []
            for val in x:
                # Handle None values specially
                if val is None:
                    idx = categories.index(None) if None in categories else -1
                else:
                    idx = categories.index(val) if val in categories else -1
                cat_indices.append(idx)
            
            for j, cat in enumerate(categories):
                idx = [i for i, val in enumerate(cat_indices) if val == j]
                if idx:
                    ax.scatter([j] * len(idx), [-y[i] for i in idx], alpha=0.6)
            
            # Create display labels for categories (convert None to "None" for display)
            display_categories = [str(cat) for cat in categories]
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels(display_categories)
        else:
            ax.scatter(x, [-yi for yi in y], alpha=0.6)
        
        ax.set_xlabel(name)
        ax.set_ylabel('C-index')
    
    plt.tight_layout()
    plt.savefig('hyperopt_results/plots/parameter_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_final_model(best_model, X_train, X_val, X_test, y_train, y_val, y_test):
    """Evaluate the final model on all datasets"""
    # Predictions
    train_pred = best_model.predict(X_train)
    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)
    
    # Calculate C-index for all sets
    c_index_train = evaluate_model(best_model, X_train, y_train)
    c_index_val = evaluate_model(best_model, X_val, y_val)
    c_index_test = evaluate_model(best_model, X_test, y_test)
    
    # Report results
    print("\nFinal Model Performance:")
    print(f"Training C-index: {c_index_train:.4f}")
    print(f"Validation C-index: {c_index_val:.4f}")
    print(f"Test C-index: {c_index_test:.4f}")
    
    # Calculate ROC-AUC if possible
    try:
        roc_auc_train = roc_auc_score(y_train['event'], train_pred)
        roc_auc_val = roc_auc_score(y_val['event'], val_pred)
        roc_auc_test = roc_auc_score(y_test['event'], test_pred)
        
        print(f"Training ROC-AUC: {roc_auc_train:.4f}")
        print(f"Validation ROC-AUC: {roc_auc_val:.4f}")
        print(f"Test ROC-AUC: {roc_auc_test:.4f}")
    except Exception as e:
        print(f"Could not calculate ROC-AUC: {str(e)}")
    
    # Save results to file
    results = {
        'train_c_index': c_index_train,
        'val_c_index': c_index_val,
        'test_c_index': c_index_test
    }
    
    pd.DataFrame([results]).to_csv('hyperopt_results/final_evaluation.csv', index=False)
    
    return results

def main():
    """Main function to run hyperparameter optimization"""
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
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = split_data_for_hyperopt(
        df,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        categorical_cols=categorical_cols,
        id_col='ID'
    )
    
    # Run hyperparameter optimization
    optimization_results = hyperparameter_optimization(
        X_train, X_val, y_train, y_val, feature_names, n_calls=50
    )
    
    # Evaluate final model
    best_model = optimization_results['best_model']
    evaluate_final_model(best_model, X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Print summary
    print("\nOptimization complete!")
    print(f"Best parameters saved to: hyperopt_results/best_params.pkl")
    print(f"Best model saved to: hyperopt_results/best_model.pkl")
    print(f"Visualization plots saved to: hyperopt_results/plots/")
    
    # Save a summary file with instructions on how to use the best parameters
    with open('hyperopt_results/README.txt', 'w') as f:
        f.write("Gradient Boosting Survival Analysis - Hyperparameter Optimization Results\n")
        f.write("=" * 70 + "\n\n")
        f.write("Best parameters:\n")
        for param, value in optimization_results['best_params'].items():
            f.write(f"{param}: {value}\n")
        f.write(f"\nBest validation C-index: {optimization_results['best_c_index']:.4f}\n\n")
        f.write("To use these parameters in your training script, update the model initialization as follows:\n\n")
        f.write("model = GradientBoostingSurvivalAnalysis(\n")
        for i, (param, value) in enumerate(optimization_results['best_params'].items()):
            comma = "," if i < len(optimization_results['best_params']) - 1 else ""
            f.write(f"    {param}={value}{comma}\n")
        f.write("    random_state=42\n")
        f.write(")\n")

if __name__ == "__main__":
    main()