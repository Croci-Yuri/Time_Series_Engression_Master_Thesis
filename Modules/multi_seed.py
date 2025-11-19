#################################################################################################################
                                # Import necessary libraries and modules
#################################################################################################################

# Import necessasry libraries
import os, sys, torch
import numpy as np
import pandas as pd
from datetime import datetime
import gc
from torch import nn
import time
import traceback
from tqdm.auto import tqdm

# Set working directory
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary modules
from Modules.seed import seed_setting

# Import models
from Models.engression import Engression
from Models.h_engression import H_Engression  
from Models.sequential_engression import Sequential_Engression
from Models.h_sequential_engression import H_Sequential_Engression
from Models.deterministic import MLP, Sequential_MLP

# Import training and evaluation functions
from Modules.training_evaluation import *
from Modules.training_evaluation_deterministic import *
from Modules.loss import multi_energy_loss
from Modules.Data_Preparation import *


#################################################################################################################
                                        # Helper Functions to build dataloader
#################################################################################################################

def _build_both_loaders(df, lags, val_size=0.2, batch_size=None):
    """
    Build both loader formats once and return them together with scalers.
    
    Creates sequence format loaders for sequential models and flat format loaders for MLP models.
    Applies standardization using training set statistics.
    """
    train_seq, val_seq, scaler_X, scaler_y = prepare_train_val_loaders_with_lags(
        df=df, lags=lags, val_size=val_size, batch_size=batch_size, standardize=True, sequence_format=True
    )
    train_flat, val_flat, _, _ = prepare_train_val_loaders_with_lags(
        df=df, lags=lags, val_size=val_size, batch_size=batch_size, standardize=True, sequence_format=False
    )
    return {"sequence": (train_seq, val_seq), "flat": (train_flat, val_flat)}, scaler_X, scaler_y


#################################################################################################################
                            # Multi-Seed Model Evaluation Function (Enhanced)
#################################################################################################################

def evaluate_models_multiple_seeds(
    df_train_val,  # Training + validation dataset  
    df_test,       # Separate test dataset
    experiment_type='sim',  # 'sim' or 'river'
    lags=10,
    val_size=0.2,
    batch_size=256,
    seeds=None, 
    true_quantiles_df=None,  # Only for test set
    results_path=None,
    device=None
):
    """
    Evaluate multiple models across different seeds using separate train/test datasets.
    
    Trains and evaluates 6 models (MLP, Sequential_MLP, Engression, H_Engression, 
    Sequential_Engression, H_Sequential_Engression) across multiple random seeds.
    Uses experiment-specific optimal configurations from grid search results.
    
    Args:
        df_train_val: DataFrame for training and validation (will be split chronologically)
        df_test: Separate DataFrame for testing
        experiment_type: Experiment type - 'sim' for simulation, 'river' for river discharge
        lags: Number of lags for data preparation
        val_size: Proportion of df_train_val to use for validation
        batch_size: Batch size for DataLoader (None for full batch)
        seeds: List of seeds to use (default: [1,2,3,4,5,6,7,8,9,10])
        true_quantiles_df: DataFrame with true quantile values [y_t, q10, q50, q90] for TEST SET ONLY
        results_path: Path to save results CSV file
        device: Device for computation
    
    Returns:
        pd.DataFrame: Comprehensive results across all seeds and models
        dict: All predictions for each model-seed combination with enhanced metadata
    """
    
    # Default parameters
    if seeds is None:
        seeds = list(range(1, 11))
        
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Validate experiment type for selection of respective optimal model
    if experiment_type not in ['sim', 'river']:
        raise ValueError("experiment_type must be 'sim' or 'river'")
    
    #################################################################################################################
                                    # Model Configurations by Experiment Type
    #################################################################################################################
    
    if experiment_type == 'sim':
        # Simulation-optimized configurations from grid search results
        model_configs = {
            'mlp': {
                'class': MLP,
                'kwargs': {
                    'input_size': lags * len(df_train_val.columns),
                    'mlp_hidden_size': 64,  
                    'output_size': 1
                },
                'sequential': False,
                'stochastic': False,
                'patience': 10,
                'lr': 5e-4,
                'weight_decay': 1e-4
            },
            'sequential_mlp': {
                'class': Sequential_MLP,
                'kwargs': {
                    'input_size': len(df_train_val.columns),
                    'gru_hidden_size': 64, 
                    'output_size': 1,
                    'pooling': 'static_softmax', 
                    'sequence_length': lags,
                    'mlp_hidden_size': [32, 32] 
                },
                'sequential': True,
                'stochastic': False,
                'patience': 10,
                'lr': 5e-4,
                'weight_decay': 1e-4
            },
            'engression': {
                'class': Engression,
                'kwargs': {
                    'input_dim': lags * len(df_train_val.columns),
                    'mlp_hidden_dim': 64, 
                    'mlp_num_layers': 1,
                    'mlp_sigma': 1,  
                    'mlp_noise_dim': 100,
                    'output_dim': 1
                },
                'sequential': False,
                'stochastic': True,
                'patience': 15,
                'lr': 5e-4,
                'weight_decay': 1e-4
            },
            'h_engression': {
                'class': H_Engression,
                'kwargs': {
                    'input_dim': lags * len(df_train_val.columns),
                    'mlp_hidden_dim': 64,  
                    'mlp_noise_dim': 100,
                    'mlp_noise_representation': 'vector',  
                    'output_dim': 1
                },
                'sequential': False,
                'stochastic': True,
                'patience': 15,
                'lr': 5e-4,
                'weight_decay': 0  
            },
            'sequential_engression': {
                'class': Sequential_Engression,
                'kwargs': {
                    'input_dim': len(df_train_val.columns),
                    'gru_hidden_size': 64,  
                    'pooling': 'static_softmax',  
                    'sequence_length': lags,
                    'mlp_hidden_dim': 64,  
                    'mlp_sigma': 1,
                    'mlp_noise_dim': 100,
                    'output_dim': 1
                },
                'sequential': True,
                'stochastic': True,
                'patience': 15,
                'lr': 1e-3,
                'weight_decay': 0  
            },
            'h_sequential_engression': {
                'class': H_Sequential_Engression,
                'kwargs': {
                    'input_dim': len(df_train_val.columns),
                    'gru_hidden_size': [64, 64],  
                    'pooling': 'static_softmax',
                    'sequence_length': lags,  
                    'mlp_hidden_dim': 64,  
                    'mlp_noise_dim': 100,
                    'mlp_noise_representation': 'vector',  
                    'output_dim': 1
                },
                'sequential': True,
                'stochastic': True,
                'patience': 15,
                'lr': 5e-4,
                'weight_decay': 0  
            }
        }
    
    elif experiment_type == 'river':
        # River discharge-optimized configurations from grid search results
        model_configs = {
            'mlp': {
                'class': MLP,
                'kwargs': {
                    'input_size': lags * len(df_train_val.columns),
                    'mlp_hidden_size': 64,  
                    'output_size': 1
                },
                'sequential': False,
                'stochastic': False,
                'patience': 10,
                'lr': 5e-4,
                'weight_decay': 0  
            },
            'sequential_mlp': {
                'class': Sequential_MLP,
                'kwargs': {
                    'input_size': len(df_train_val.columns),
                    'gru_hidden_size': 128,  
                    'output_size': 1,
                    'pooling': 'static_softmax',  
                    'sequence_length': lags,
                    'mlp_hidden_size': [64, 64]  
                },
                'sequential': True,
                'stochastic': False,
                'patience': 10,
                'lr': 1e-3,
                'weight_decay': 1e-4
            },
            'engression': {
                'class': Engression,
                'kwargs': {
                    'input_dim': lags * len(df_train_val.columns),
                    'mlp_hidden_dim': [64, 64], 
                    'mlp_num_layers': 2,
                    'mlp_sigma': 1,  
                    'mlp_noise_dim': 100,
                    'output_dim': 1
                },
                'sequential': False,
                'stochastic': True,
                'patience': 15,
                'lr': 5e-4,
                'weight_decay': 1e-4
            },
            'h_engression': {
                'class': H_Engression,
                'kwargs': {
                    'input_dim': lags * len(df_train_val.columns),
                    'mlp_hidden_dim': 64,  
                    'mlp_noise_dim': 100,
                    'mlp_noise_representation': 'vector',  
                    'output_dim': 1
                },
                'sequential': False,
                'stochastic': True,
                'patience': 15,
                'lr': 5e-4,
                'weight_decay': 1e-4
            },
            'sequential_engression': {
                'class': Sequential_Engression,
                'kwargs': {
                    'input_dim': len(df_train_val.columns),
                    'gru_hidden_size': 128,  
                    'pooling': 'static_softmax',  
                    'sequence_length': lags,
                    'mlp_hidden_dim': 64,  
                    'mlp_sigma': 1,
                    'mlp_noise_dim': 100,
                    'output_dim': 1
                },
                'sequential': True,
                'stochastic': True,
                'patience': 15,
                'lr': 1e-3,
                'weight_decay': 0 
            },
            'h_sequential_engression': {
                'class': H_Sequential_Engression,
                'kwargs': {
                    'input_dim': len(df_train_val.columns),
                    'gru_hidden_size': 128,  
                    'pooling': 'static_softmax',  
                    'sequence_length': lags,
                    'mlp_hidden_dim': 64,  
                    'mlp_noise_dim': 100,
                    'mlp_noise_representation': 'vector',  
                    'output_dim': 1
                },
                'sequential': True,
                'stochastic': True,
                'patience': 15,
                'lr': 1e-3,
                'weight_decay': 0 
            }
        }
    
    # Initialize storage for results and predictions
    results = []
    all_predictions = {}
    
    # Calculate total configurations for progress tracking
    total_configs = len(model_configs) * len(seeds)
    print(f"Starting evaluation across {len(seeds)} seeds and {len(model_configs)} models...")
    print(f"Training data: {(len(df_train_val))*(1-val_size)} obs, Validation data: {len(df_train_val)*val_size} obs, Test data: {len(df_test)} obs")

    
    # Progress bar setup
    pbar = tqdm(
        total=total_configs,
        desc="Multi-seed evaluation",
        unit="config",
        leave=True,
        dynamic_ncols=True,
        mininterval=0.5
    )
    
    #################################################################################################################
                                        # Main Evaluation Loop
    #################################################################################################################
    
    for seed in seeds:
        print(f"\n=== Processing Seed {seed} ===")
        seed_setting(seed)
        
        # Build train-val loaders from df_train_val
        loaders_both, scaler_X, scaler_y = _build_both_loaders(
            df=df_train_val, 
            lags=lags, 
            val_size=val_size, 
            batch_size=batch_size
        )
        
        # Build test loaders from df_test using the same scalers
        test_loader_seq = build_eval_loader_with_lags(
            df=df_test,
            lags=lags,
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            batch_size=batch_size,
            sequence_format=True
        )
        
        test_loader_flat = build_eval_loader_with_lags(
            df=df_test,
            lags=lags, 
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            batch_size=batch_size,
            sequence_format=False
        )
        
        # Store test data metadata
        test_data_length = len(test_loader_seq.dataset)
        
        # Process true quantiles
        true_quantiles_aligned = None
        if true_quantiles_df is not None:
            try:
                test_data_length = len(test_loader_seq.dataset)

                # Since true_quantiles_df is only for test set, we need to account for the lags removed during the data preparation
                # of teh input space.
                # The test data starts from index 'lags' after create_lagged_features drops first 'lags' rows
                
                if len(true_quantiles_df) >= test_data_length + lags:
                    # Skip first 'lags' rows to align with processed test data
                    aligned_df = true_quantiles_df.iloc[lags:lags+test_data_length].reset_index(drop=True)
                    true_quantiles_aligned = {
                        'q10': aligned_df['q10'].values if 'q10' in aligned_df.columns else None,
                        'q50': aligned_df['q50'].values if 'q50' in aligned_df.columns else None,
                        'q90': aligned_df['q90'].values if 'q90' in aligned_df.columns else None,
                        'y_t': aligned_df['y_t'].values if 'y_t' in aligned_df.columns else None  # Store true values
                    }
                else:
                    print(f"Warning: true_quantiles_df has {len(true_quantiles_df)} rows, need at least {test_data_length + lags}")
                    
            except Exception as e:
                print(f"Error processing true quantiles: {e}")
        
        # Evaluate each model configuration
        for model_name, config in model_configs.items():
            start_time = time.perf_counter()
            
            try:
                # Select appropriate loaders based on model type
                if config['sequential']:
                    train_loader, val_loader = loaders_both["sequence"]
                    test_loader = test_loader_seq
                else:
                    train_loader, val_loader = loaders_both["flat"]
                    test_loader = test_loader_flat
                
                # Initialize model and optimizer with model-specific hyperparameters
                model = config['class'](**config['kwargs']).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
                
                # Train model with early stopping
                if config['stochastic']:
                    # Train stochastic Engression-based model
                    train_history = train_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        num_epochs=1000,
                        m=2,
                        device=device,
                        early_stopping=True,
                        patience=config['patience'],
                        delta=5e-4,
                        m_validation=50,
                        verbose=False,
                        print_every=25,
                        eval_every=2
                    )
                    
                    # Evaluate stochastic model
                    eval_results = evaluate_model(
                        model=model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        scaler_y=scaler_y,
                        m=100,
                        return_preds=True,
                        upperquantile_mse=0.995,
                        coverage=0.8,
                        lower_quantile=0.1,
                        upper_quantile=0.9,
                        device=device,
                        verbose=False
                    )
                    
                else:
                    # Train deterministic MLP-based model
                    train_history = train_model_deterministic(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        num_epochs=1000,
                        device=device,
                        early_stopping=True,
                        patience=config['patience'],
                        delta=5e-4,
                        print_every=25,
                        verbose=False,
                        eval_every=2
                    )
                    
                    # Evaluate deterministic model
                    eval_results = evaluate_model_deterministic(
                        model=model,
                        train_loader=train_loader,
                        test_loader=test_loader,
                        scaler_y=scaler_y,
                        return_preds=True,
                        upperquantile_mse=0.995,
                        device=device,
                        verbose=False
                    )
                
                duration = time.perf_counter() - start_time
                
                #################################################################################################################
                                                # Compile Results for Current Model-Seed Combination
                #################################################################################################################
                
                # Base metrics common to all models
                result_row = {
                    'seed': seed,
                    'model_name': model_name,
                    'best_epoch': train_history.get('best_epoch', None),
                    'best_val_loss': train_history.get('best_val', None),
                    'best_train_at_best': train_history.get('best_train_at_best', None),
                    'duration_sec': duration,
                    'mse': eval_results['mse'],
                    'rmse': np.sqrt(eval_results['mse']),
                    'mae': eval_results['mae_mean'],
                    'upperquantile_mse': eval_results.get('upperquantile_mse', np.nan),
                    'upperquantile_rmse': np.sqrt(eval_results['upperquantile_mse']) if not np.isnan(eval_results.get('upperquantile_mse', np.nan)) else np.nan,
                    'upperquantile_count': eval_results.get('upperquantile_count', 0),
                    'status': 'ok'
                }
                
                # Add stochastic-specific metrics for Engression models
                if config['stochastic']:
                    result_row.update({
                        'energy_loss': eval_results['energy_loss'],
                        's1': eval_results['s1'],
                        's2': eval_results['s2'],
                        'picp_80': eval_results['coverage'],
                        'sharpness_80': eval_results['sharpness']
                    })
                    
                    # Compute quantile errors if true quantiles provided (simulation only)
                    if true_quantiles_aligned is not None:
                        if 'y_pred_lower' in eval_results and true_quantiles_aligned['q10'] is not None:
                            q10_mse = np.mean((true_quantiles_aligned['q10'] - eval_results['y_pred_lower'])**2)
                            result_row['q10_mse'] = q10_mse
                            result_row['q10_rmse'] = np.sqrt(q10_mse)
                        
                        if 'y_pred_median' in eval_results and true_quantiles_aligned['q50'] is not None:
                            q50_mse = np.mean((true_quantiles_aligned['q50'] - eval_results['y_pred_median'])**2)
                            result_row['q50_mse'] = q50_mse
                            result_row['q50_rmse'] = np.sqrt(q50_mse)
                        
                        if 'y_pred_upper' in eval_results and true_quantiles_aligned['q90'] is not None:
                            q90_mse = np.mean((true_quantiles_aligned['q90'] - eval_results['y_pred_upper'])**2)
                            result_row['q90_mse'] = q90_mse
                            result_row['q90_rmse'] = np.sqrt(q90_mse)
                    
                    # Fill missing quantile metrics with NaN
                    for q in ['q10', 'q50', 'q90']:
                        if f'{q}_mse' not in result_row:
                            result_row[f'{q}_mse'] = np.nan
                            result_row[f'{q}_rmse'] = np.nan
                
                else:
                    # Fill with NaN for deterministic models (no distributional metrics)
                    result_row.update({
                        'energy_loss': np.nan,
                        's1': np.nan,
                        's2': np.nan,
                        'picp_80': np.nan,
                        'sharpness_80': np.nan,
                        'q10_mse': np.nan,
                        'q10_rmse': np.nan,
                        'q50_mse': np.nan,
                        'q50_rmse': np.nan,
                        'q90_mse': np.nan,
                        'q90_rmse': np.nan
                    })
                
                results.append(result_row)
                
                # Prediction storage
                pred_key = f"{model_name}_seed{seed}"
                
                # Core predictions
                prediction_data = {
                    'y_true': eval_results['y_true'],
                    'y_pred': eval_results.get('y_pred', eval_results.get('y_pred_mean')),
                    'seed': seed,
                    'model_name': model_name,
                    'data_length': len(eval_results['y_true']),
                    'scaler_info': {
                        'y_mean': float(scaler_y.mean_[0]) if scaler_y else None,
                        'y_scale': float(scaler_y.scale_[0]) if scaler_y else None
                    }
                }
                
                # Add quantile predictions for stochastic models
                if config['stochastic']:
                    if 'y_pred_lower' in eval_results:
                        prediction_data['y_pred_q10'] = eval_results['y_pred_lower']
                    if 'y_pred_median' in eval_results:
                        prediction_data['y_pred_median'] = eval_results['y_pred_median']
                    if 'y_pred_upper' in eval_results:
                        prediction_data['y_pred_q90'] = eval_results['y_pred_upper']
                
                # Add true quantiles if available
                if true_quantiles_aligned is not None:
                    prediction_data['true_quantiles'] = {
                        'q10': true_quantiles_aligned['q10'],
                        'q50': true_quantiles_aligned['q50'], 
                        'q90': true_quantiles_aligned['q90'],
                        'y_t': true_quantiles_aligned['y_t']
                    }
                
                all_predictions[pred_key] = prediction_data
                
                # Update progress display
                last_postfix = f"Last: {model_name} | RMSE={np.sqrt(eval_results['mse']):.4f} | {duration:.1f}s"
                
            except Exception as e:
                # Detailed error reporting for debugging
                print(f"\n{'='*60}")
                print(f"ERROR in {model_name} with seed {seed}")
                print(f"Error: {repr(e)}")
                print("Full traceback:")
                print(traceback.format_exc())
                print(f"{'='*60}\n")
                
                # Add error row to results
                error_row = {
                    'seed': seed,
                    'model_name': model_name,
                    'best_epoch': None,
                    'best_val_loss': None,
                    'best_train_at_best': None,
                    'duration_sec': None,
                    'status': 'error',
                    'error': str(e)
                }

                # Fill metrics with NaN
                for key in ['mse', 'rmse', 'mae', 'energy_loss', 's1', 's2', 'picp_80', 'sharpness_80',
                           'upperquantile_mse', 'upperquantile_rmse', 'upperquantile_count',
                           'q10_mse', 'q10_rmse', 'q50_mse', 'q50_rmse', 'q90_mse', 'q90_rmse']:
                    
                    error_row[key] = np.nan

                results.append(error_row)
                last_postfix = f"{model_name} | ERROR"
                
            finally:
                # Memory cleanup after each model
                if 'model' in locals():
                    del model
                if 'optimizer' in locals():
                    del optimizer
                
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
                # Update progress bar
                pbar.update(1)
                pbar.set_postfix_str(last_postfix)
    
    pbar.close()
    
    #################################################################################################################
                                        # Save Results and Return
    #################################################################################################################
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Results and predictions saving if path provided
    if results_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{results_path}_multiseed_evaluation_{timestamp}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
        
        # Predictions saving with comprehensive metadata
        pred_filename = f"{results_path}_predictions_{timestamp}.pkl"
        prediction_metadata = {
            'predictions': all_predictions,
            'experiment_info': {
                'experiment_type': experiment_type,
                'lags': lags,
                'val_size': val_size,
                'seeds': seeds,
                'test_data_length': test_data_length,
                'timestamp': timestamp
            },
            'model_configs': {name: {k: v for k, v in config.items() if k != 'class'} 
                            for name, config in model_configs.items()}
        }
        
        import pickle
        with open(pred_filename, 'wb') as f:
            pickle.dump(prediction_metadata, f)
        print(f"Predictions saved to: {pred_filename}")
    
    return results_df, all_predictions



#################################################################################################################
                                    # Summary Statistics Function
#################################################################################################################

def compute_summary_statistics(results_df):
    """
    Compute mean and std across seeds for each model.
    
    Provides performance summary statistics across random seeds to assess
    model performance and variability.
    """
    
    # Metrics to summarize
    numeric_columns = ['mse', 'rmse', 'mae', 'energy_loss', 's1', 's2', 'picp_80', 'sharpness_80',
                      'upperquantile_mse', 'upperquantile_rmse', 'upperquantile_count',
                      'q10_mse', 'q10_rmse', 'q50_mse', 'q50_rmse', 'q90_mse', 'q90_rmse',
                      'best_val_loss', 'duration_sec']
    
    summary_stats = []
    
    # Compute statistics for each model
    for model_name in results_df['model_name'].unique():
        model_data = results_df[results_df['model_name'] == model_name]
        
        stats_row = {'model_name': model_name}
        
        # Calculate mean and std for each metric
        for col in numeric_columns:
            if col in model_data.columns:
                values = model_data[col].dropna()
                if len(values) > 0:
                    stats_row[f'{col}_mean'] = values.mean()
                    stats_row[f'{col}_std'] = values.std()
                else:
                    stats_row[f'{col}_mean'] = np.nan
                    stats_row[f'{col}_std'] = np.nan
        
        summary_stats.append(stats_row)
    
    return pd.DataFrame(summary_stats)

