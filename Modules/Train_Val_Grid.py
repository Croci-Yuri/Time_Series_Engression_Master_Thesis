#################################################################################################################
                                # Import libraries, modules and models
#################################################################################################################

# Import pacakges
import os, sys, time, json, traceback
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from tqdm.auto import tqdm
from itertools import product
import gc

# Set current and repository working directory
current_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)


# Import models
from Models.engression import Engression
from Models.sequential_engression import Sequential_Engression
from Models.h_engression import H_Engression
from Models.h_sequential_engression import H_Sequential_Engression
from Models.deterministic import MLP, Sequential_MLP

# Import Modules
from Modules.training_evaluation import train_model
from Modules.training_evaluation_deterministic import train_model_deterministic
from Modules.Data_Preparation import *
from Modules.seed import seed_setting



#################################################################################################################
                                    # Single training-validation gridsearch              
#################################################################################################################

## Model registry ##
####################

MODEL_REGISTRY = {
    "mlp":                     {"sequential": False, "stochastic": False, "class": MLP},
    "sequential_mlp":          {"sequential": True,  "stochastic": False, "class": Sequential_MLP},
    "engression":              {"sequential": False, "stochastic": True,  "class": Engression},
    "sequential_engression":   {"sequential": True,  "stochastic": True,  "class": Sequential_Engression},
    "h_engression":            {"sequential": False, "stochastic": True,  "class": H_Engression},
    "h_sequential_engression": {"sequential": True,  "stochastic": True,  "class": H_Sequential_Engression},
}


## Train and Validation Dataloader ##
#####################################

def _build_both_loaders(df, lags, val_size=0.2, batch_size=None, standardize=True):
    """
    Build both loader formats once and return them together with scaler_y for metrics.
    """
    train_seq, val_seq, scaler_X, scaler_y = prepare_train_val_loaders_with_lags(
        df=df, lags=lags, val_size=val_size, batch_size=batch_size, standardize=standardize, sequence_format=True
    )
    train_flat, val_flat, _, _ = prepare_train_val_loaders_with_lags(
        df=df, lags=lags, val_size=val_size, batch_size=batch_size, standardize=standardize, sequence_format=False
    )
    return {"sequence": (train_seq, val_seq), "flat": (train_flat, val_flat)}, scaler_y


## Selection of the correct DataLoader pair ##
##############################################

def _pick_loader_pair(loaders_both, sequential):
    return loaders_both["sequence"] if sequential else loaders_both["flat"]


##################################
## TRAIN_VALIDATION GRID SEARCH ##
##################################

def run_chronological_grid_search(
    configs,
    df,
    lags,
    val_size=0.2,
    batch_size=None,
    standardize=True,
    device=None,
    result_csv_path=None,
    experiment_name=None,
    seed=None,
):
    """
Run a chronological single split grid search with early stopping.

Args:
    configs (list[dict]): Model families and grids. Each item must include:
        - name (str): Key in MODEL_REGISTRY.
        - param_grid (list[dict]): List of constructor kwargs for the model.
        - learning_rate (float): Learning rate for AdamW.
        - weight_decay (float): Weight decay for AdamW.
        - train_kw (dict): Must include 'num_epochs'. Optional trainer args.
    df (pd.DataFrame): Time ordered data with target in the last column.
    lags (int): Number of lagged steps used to build inputs.
    val_size (float): Fraction of the series reserved for validation.
    batch_size (int or None): Batch size. If None, full batch is used.
    standardize (bool): If True, standardize using train statistics.
    device (str or None): 'cuda' or 'cpu'. If None, auto select.
    result_csv_path (str or None): If set, save the results DataFrame to CSV.
    experiment_name (str or None): If set, append to the CSV filename.
    seed (int or None): Random seed for reproducibility.

Returns:
    pd.DataFrame: One row per configuration with:
        - 'model': Model name.
        - 'params': Parameter dictionary used to build the model.
        - 'learning_rate': Learning rate used.
        - 'weight_decay': Weight decay used.
        - 'best_epoch': Epoch index achieving the best validation loss.
        - 'best_val': Best validation loss from early stopping.
        - 'best_train_at_best': Training loss at best_epoch.
        - 'duration_sec': Wall time in seconds for train plus validation.
        - 'status': 'ok' or 'error'.

    """

    # Select device
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Reproducibility
    if seed is not None:
        seed_setting(random_state=seed)

    # Build loaders once (sequence and flat)
    loaders_both, _ = _build_both_loaders(df, lags, val_size, batch_size, standardize)

    # Bookkeeping
    rows = []
    total = sum(len(cfg["param_grid"]) for cfg in configs)
    run_count = 0
    print(f"Running {total} configurations on a single chronological split...")

    # Single progress bar for all configs
    pbar = tqdm(
        total=total,
        desc="Chronological grid search",
        unit="config",
        leave=True,          
        dynamic_ncols=True,
        mininterval=0.5
    )

    # Loop over model families
    for cfg in configs:
        name = cfg["name"].lower()
        if name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model name '{name}'. Choose among {list(MODEL_REGISTRY.keys())}")

        # Model facts
        facts = MODEL_REGISTRY[name]
        ModelClass = facts["class"]
        sequential = facts["sequential"]
        stochastic = facts["stochastic"]

        # Training settings
        lr = cfg["learning_rate"]
        wd = cfg["weight_decay"]
        train_kw = dict(cfg.get("train_kw", {}))

        # Pick input format
        train_loader, val_loader = _pick_loader_pair(loaders_both, sequential)

        # Iterate hyperparameter combinations
        for model_kw in cfg["param_grid"]:
            start = time.perf_counter()
            last_postfix = f"{name} | starting..."
            try:
                # Model and optimizer (always AdamW)
                model = ModelClass(**model_kw).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

                # Trainer kwargs
                tk = {**train_kw}
                if "num_epochs" not in tk:
                    raise ValueError("train_kw must include 'num_epochs'.")
                num_epochs = tk.pop("num_epochs")

                # Train by family
                if stochastic:
                    history = train_model(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        num_epochs=num_epochs,
                        **tk
                    )
                else:
                    history = train_model_deterministic(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        num_epochs=num_epochs,
                        device=device,
                        **tk
                    )

                # Minimal row
                dur = time.perf_counter() - start
                row = {
                    "model": name,
                    "params": model_kw,
                    "learning_rate": lr,
                    "weight_decay": wd,
                    "best_epoch": history.get("best_epoch"),
                    "best_val": history.get("best_val"),
                    "best_train_at_best": history.get("best_train_at_best"),
                    "duration_sec": dur,
                    "status": "ok",
                }
                rows.append(row)

                # Compact status string
                bv = row["best_val"]
                dsec = row["duration_sec"]
                bv_str = f"{bv:.6f}" if bv is not None else "nan"
                last_postfix = f"Last: {name} | best={bv_str} | {dsec:.0f}s"

            except Exception as e:
                # Print detailed error information immediately
                print(f"\n{'='*60}")
                print(f"ERROR in {name} with params: {model_kw}")
                print(f"Learning rate: {lr}, Weight decay: {wd}")
                print(f"Error: {repr(e)}")
                print("Full traceback:")
                print(traceback.format_exc())
                print(f"{'='*60}\n")
                
                # Store minimal error info in CSV
                rows.append({
                    "model": name,
                    "params": model_kw,
                    "learning_rate": lr,
                    "weight_decay": wd,
                    "best_epoch": None,
                    "best_val": None,
                    "best_train_at_best": None,
                    "duration_sec": None,
                    "status": "error",
                })
                
                last_postfix = f"{name} | ERROR"

            finally:
                # Progress + cleaning
                run_count += 1
                pbar.update(1)
                pbar.set_postfix_str(last_postfix)
                
                # Clean up system after every model to avoid overload GPU, CPU and RAM
                if 'model' in locals():
                    del model
                if 'optimizer' in locals():
                    del optimizer
                if 'history' in locals():
                    del history

                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
                # Periodic system break
                if run_count % 10 == 0:
                    print(f"\nTaking brief system break after {run_count} configs...")
                    time.sleep(5)


    pbar.close()

    # Assemble results and save
    df_res = pd.DataFrame(rows)

    if result_csv_path:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        base_name = result_csv_path.replace('.csv', '') if result_csv_path.endswith('.csv') else result_csv_path
        
        if experiment_name:
            if seed is not None:
                out = f"{base_name}_{experiment_name}_seed{seed}_{ts}.csv"
            else:
                out = f"{base_name}_{experiment_name}_{ts}.csv"
        else:
            if seed is not None:
                out = f"{base_name}_seed{seed}_{ts}.csv"
            else:
                out = f"{base_name}_{ts}.csv"
        
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        df_res.to_csv(out, index=False)
        print(f"Saved results to: {out}")

        return df_res


#################################################################################################################
                                        # Expand paramsgrid search
#################################################################################################################

def expand_param_grid(base_config, search_space):

    """Generate all parameter combinations from base config and search space."""

    keys, values = zip(*search_space.items())
    return [dict(base_config, **dict(zip(keys, v))) for v in product(*values)]
