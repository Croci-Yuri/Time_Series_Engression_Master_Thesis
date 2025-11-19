#################################################################################################################
                                     # Import libraries
#################################################################################################################


import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch


#################################################################################################################
                        # Function to build lagged features for time series data
#################################################################################################################

def create_lagged_features(df, lags):
    """
    Create lagged features for the regressors and response variable (horizon = 1).

    Args:
        df (pd.DataFrame): Original DataFrame where all columns except the last are regressors,
                           and the last column is the response variable.
        lags (int): Number of lagged steps to create.

    Returns:
        pd.DataFrame: DataFrame containing lagged regressors, lagged response variable,
                      and the original response variable.
    """

    # Response column
    response_col = df.columns[-1]  
    
    # Build lagged features
    lagged_features = {
        f'{col}_lag_{lag}': df[col].shift(lag)
        for lag in range(1, lags + 1)
        for col in df.columns
    }

    lagged_df = pd.DataFrame(lagged_features, index=df.index)

    # Stack response to lagged covaraiates
    lagged_df[response_col] = df[response_col]

    # Drop NaN values caused by shifting
    lagged_df.dropna(inplace=True)

    return lagged_df




#################################################################################################################
                # Prepare dataloaders with lagged features and standardization for time series data
#################################################################################################################


def prepare_train_val_loaders_with_lags(df, lags, val_size=0.2, batch_size=None, standardize=True, sequence_format=True):
    """
    Split into train and validation sets chronologically, standardize using train scalers,
    and build lagged dataloaders.

    Args:
        df (pd.DataFrame): DataFrame with regressors and response (last column).
        lags (int): Number of lagged steps.
        val_size (float): Proportion of data for validation.
        batch_size (int): Batch size for the DataLoader (default full batch).
        standardize (bool): Whether to apply standard scaling.
        sequence_format (bool): Return sequences [N, lags, D] or flat inputs.

    Returns:
        train_loader, val_loader, scaler_X, scaler_y
    """

    # Separate covariates from response
    regressor_cols = df.columns[:-1]
    response_col = df.columns[-1]
    n_features = len(df.columns)

    # Chronological split
    train_size = int((1 - val_size) * len(df))
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:].copy()

    # Standardize
    if standardize:
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        train_df[regressor_cols] = scaler_X.fit_transform(train_df[regressor_cols])
        train_df[response_col] = scaler_y.fit_transform(train_df[response_col].values.reshape(-1, 1))

        val_df[regressor_cols] = scaler_X.transform(val_df[regressor_cols])
        val_df[response_col] = scaler_y.transform(val_df[response_col].values.reshape(-1, 1))
    else:
        scaler_X = scaler_y = None

    # Build lagged features
    train_lagged = create_lagged_features(train_df, lags)
    val_lagged = create_lagged_features(val_df, lags)

    # Split X and y
    X_train = train_lagged.iloc[:, :-1].values
    y_train = train_lagged.iloc[:, -1].values.reshape(-1, 1)
    X_val = val_lagged.iloc[:, :-1].values
    y_val = val_lagged.iloc[:, -1].values.reshape(-1, 1)

    # Format input space into Tensor shape (n, s, d_x+1)
    if sequence_format:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).view(-1, lags, n_features)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).view(-1, lags, n_features)
    
    # Format input space into Tensor shape (n, s * (d_x+1))
    else:
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    # Format response into Tensor shape (n,1)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

    # Build loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    if batch_size is None:
        batch_size = len(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler_X, scaler_y




#################################################################################################################
                 # Build evaluation loader on test set using existing scalers computed on train-val set
#################################################################################################################

def build_eval_loader_with_lags(df, lags, scaler_X, scaler_y, batch_size=None, sequence_format=True):
    """
    Apply given scalers to a test set, build lagged features, and return a DataLoader.

    Args:
        df (pd.DataFrame): Test DataFrame with regressors and response (last column).
        lags (int): Number of lagged steps.
        scaler_X (StandardScaler): Scaler fitted on training features.
        scaler_y (StandardScaler): Scaler fitted on training response.
        batch_size (int): Batch size for the DataLoader (default full batch).
        sequence_format (bool): Return sequences [N, lags, D] or flat inputs.

    Returns:
        test_loader
        
    """

    # Separate covariates from response
    regressor_cols = df.columns[:-1]
    response_col = df.columns[-1]
    n_features = len(df.columns)

    # Apply standardization
    test_df = df.copy()
    test_df[regressor_cols] = scaler_X.transform(test_df[regressor_cols])
    test_df[response_col] = scaler_y.transform(test_df[response_col].values.reshape(-1, 1))

    # Build dataframe lagged feature
    test_lagged = create_lagged_features(test_df, lags)

    # Separate covariates from response
    X_test = test_lagged.iloc[:, :-1].values
    y_test = test_lagged.iloc[:, -1].values.reshape(-1, 1)
    
    # Format dataframe into Tensor shape (n,s,d_x+1)
    if sequence_format:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).view(-1, lags, n_features)
    
    # Format dataframe into Tensor shape (n, s * (d_x+1))
    else:
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # Format response into Tensor shape (n,1)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    if batch_size is None:
        batch_size = len(y_test_tensor)

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader