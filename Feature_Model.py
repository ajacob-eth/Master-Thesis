import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mtick

import datetime
import time
import math
import os
import json
from itertools import combinations

from laplace import Laplace

import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, SGD, SparseAdam
import torch.distributions as dist
import joblib

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, brier_score_loss, log_loss, roc_auc_score
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, RationalQuadratic
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import calibration_curve

from imblearn.over_sampling import SMOTE

from scipy.optimize import approx_fprime
from scipy.linalg import inv

class NN_Classifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=3, output_dim=1, dropout_rate=None):
        """
        Initialize neural network model.

        Args:
        input_dim (int): Number of input features
        hidden_dim (int): Dimension of each hidden layer
        output_dim (int): Number of output units
        dropout_rate (float): Dropout rate of any dropout layer. Must be between 0 and 1 or None. 

        Returns:
        None
        """

        super(NN_Classifier, self).__init__()
        self.hidden_layer1 = nn.Linear(input_dim, hidden_dim, dtype=torch.float64)
        self.dropoutrate = dropout_rate

        if dropout_rate is not None:
            if not (0 <= dropout_rate <= 1):
                raise ValueError("Dropout rate must be between 0 and 1")
            self.dropout1 = nn.Dropout(p=dropout_rate)

        self.output_layer = nn.Linear(hidden_dim, output_dim, dtype=torch.float64)
        self.apply(self.initialize_weights)

    def initialize_weights(self, layer):
        """
        Initialize weights using Xavier (Glorot) initialization.

        Args:
            layer (nn.Module): The layer to initialize.

        Returns:
            None
        """
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)            

    def forward(self, input):
        """
        Forward pass of the neural network.

        Args:
            input (torch.Tensor): Input data.

        Returns:
            torch.Tensor: Output of the neural network.
        """
        out = self.hidden_layer1(input)
        out = torch.tanh(out)

        if self.dropoutrate is not None:
            out = self.dropout1(out)

        out = self.output_layer(out)
        out = torch.sigmoid(out)

        return out


def benchmark_clustering(ltv_percentiles, ic_percentiles, ic_value, ltv_value):
    """
    Clusters IC and LTV values based on provided percentiles. Helper funtion for benchmark model.

    Args:
        ltv_percentiles (list of float): Percentiles for LTV values.
        ic_percentiles (list of float): Percentiles for IC values.
        ic_value (float): The IC value to be clustered.
        ltv_value (float): The LTV value to be clustered.

    Returns:
        str: The cluster identifier in the form of "(i, j)", or 'NaN' if either value is NaN.
    """
    # Check for NaN values in ic_value and ltv_value
    if pd.isna(ic_value) or pd.isna(ltv_value):
        return 'NaN'

    # Create intervals for IC percentiles
    ic_interval = [(ic_percentiles[i], ic_percentiles[i+1]) for i in range(len(ic_percentiles) - 1)]
    ic_interval.insert(0, (-float('inf'), ic_percentiles[0]))
    ic_interval.append((ic_percentiles[-1], float('inf')))

    # Create intervals for LTV percentiles
    ltv_interval = [(ltv_percentiles[i], ltv_percentiles[i+1]) for i in range(len(ltv_percentiles) - 1)]
    ltv_interval.insert(0, (-float('inf'), ltv_percentiles[0]))
    ltv_interval.append((ltv_percentiles[-1], float('inf')))

    # Determine the cluster for the given IC and LTV values
    for i, ic_bounds in enumerate(ic_interval):
        for j, ltv_bounds in enumerate(ltv_interval):
            if ic_bounds[0] < ic_value <= ic_bounds[1]:
                if ltv_bounds[0] < ltv_value <= ltv_bounds[1]:
                    return f"({i}, {j})"
    
    return 'NaN'  # Return 'NaN' if no cluster is found

def ebitda_cluster(x):
    """
    Helper function to apply floor and cap to EBITDA.

    Args:
        x (float): EBITDA value.

    Returns:
        float: Log-transformed EBITDA value.
    """
    if x >= 300 * 1e6:
        return np.log(300 * 1e6)
    
    elif x <= 5 * 1e6:
        return np.log(5 * 1e6)
    
    elif x <= 300 * 1e6:
        return np.log(x)
    
    else:
        return float('nan')

def LTV_cluster(x):
    """
    Helper function to apply floor and cap to LTV.

    Args:
        x (float): LTV ratio value.

    Returns:
        float: Adjusted LTV ratio for clustering.
    """
    cap = 0.85
    if x > cap:
        return cap
    elif x > 0.35:
        return x
    elif x <= 0.35:
        return 0.35
    else:
        return float('nan')

def IC_Cluster(x):
    """
    Helper function to apply floor and cap to IC.

    Args:
        x (float): IC ratio value.

    Returns:
        float: Adjusted IC ratio for clustering.
    """

    floor = 1

    assert floor < 4
    
    if x <= floor:
        return floor
    elif x <= 4:
        return x
    elif x > 4:
        return 4
    else:
        return float('nan')

def EV_Multiple_cluster(x):
    """
    Helper function to apply floor and cap to EV Multiple.

    Args:
        x (float): EV multiple value.

    Returns:
        float: Adjusted EV multiple for clustering.
    """
    floor = 6
    
    ceil = 15

    if x <= floor:
        return floor
    elif x <= ceil:
        return x
    elif x > ceil:
        return ceil
    else:
        return float('nan')
    
def Security_PreCluster(x):
    """
    Helper function to categorize security types into pre-defined clusters.

    Args:
        x (str): Security type.

    Returns:
        str: Clustered security type.
        float: NaN if the security type does not match any predefined categories.
    """
    if x in ['First Lien Senior Loan', 'Unitranche (Whole) Loan']:
        return 'First Lien or Unitranche'
    elif x in ['Second Lien Loan', 'Mezzanine']:
        return 'Second Lien or Mezzanine'
    else:
        return float('nan')

def Security_cluster(x):
    """
    Helper function to assign cluster indices to pre-categorized security types.

    Args:
        x (str): Pre-categorized security type.

    Returns:
        int: Cluster index for the given security type.
        float: NaN if the security type does not match any predefined categories.
    """
    if x == 'First Lien or Unitranche':
        return 0
    elif x == 'Second Lien or Mezzanine':
        return 1
    else:
        return float('nan')

def cluster_net_leverage(x):
    """
    Helper function to apply floor and cap to Net Leverage.

    Args:
        x (float): Net leverage value.

    Returns:
        int: Cluster index based on the net leverage value.
        float: NaN if the value does not match any predefined categories.
    """
    if x <= 4:
        return 4
    elif x <= 9:  # Ensure the correct range for the second cluster
        return x
    elif x > 9:
        return 9
    else:
        return float('nan')

def base_case_cum_net_return(fprs, tprs, thresholds=np.linspace(0.01, 0.99, 99, endpoint=True), 
                             loan_life=3, default_rate=0.05):
    """
    Calculate a metric comparing different classifiers based on loan performance.

    Args:
        fprs (array-like): Array of false positive rates (FPR).
        tprs (array-like): Array of true positive rates (TPR).
        thresholds (array-like, optional): Threshold values for classification.
        loan_life (float, optional): Loan maturity in years.
        default_rate (float, optional): Overall default rate.

    Returns:
        tuple: A tuple containing:
            - float: Difference from the base case portfolio return (in bps).
            - float: Optimal threshold value.
            - float: False positive rate corresponding to the optimal threshold.
            - float: True positive rate corresponding to the optimal threshold.
    """
    Num_of_loans = 100
    overall_default_rate = default_rate
    bad_loans = np.ceil(Num_of_loans * overall_default_rate)
    good_loans = Num_of_loans - bad_loans

    r_per_deal = 0.1
    recovery_rate = 0.5
    r_alternative = 0.05
    Maturity = loan_life

    # Calculate annualized base case portfolio return
    annualized_base_case_portfolio_return = ((good_loans * (1 + r_per_deal)**Maturity 
                                              + bad_loans * recovery_rate * (1 + r_per_deal)**Maturity) 
                                             / Num_of_loans)**(1/loan_life) - 1

    cum_net_returns = []

    # Calculate adjusted portfolio return for each threshold
    for fpr, tpr in zip(fprs, tprs):
        TPR = tpr
        FPR = fpr
        TP = np.round(TPR * bad_loans)
        FP = np.round(FPR * good_loans)
        
        if not np.isnan(TP) and not np.isnan(FP):
            adj_portfolio_return = (good_loans - FP) * (1 + r_per_deal)**Maturity \
                                    + (bad_loans - TP) * recovery_rate * (1 + r_per_deal)**Maturity \
                                    + (TP + FP) * (1 + r_alternative)**Maturity
            adj_portfolio_return /= Num_of_loans
            cum_net_returns.append(adj_portfolio_return)
        else:
            cum_net_returns.append(float('nan'))

    # Find index of maximum adjusted portfolio return
    ind = np.argmax(cum_net_returns)
    
    # Return tuple with metrics at optimal threshold
    return (10000 * ((cum_net_returns[ind])**(1/loan_life) - 1 - annualized_base_case_portfolio_return), 
            thresholds[ind], fprs[ind], tprs[ind])

def pre_processing(df0, features, simply_clustered=False, scaling=False, trained_scaler=None, scaler=StandardScaler(), print_weights=False):
    """
    Processes the input dataframe and returns a cleaned and preprocessed dataframe.

    Args:
        df0 (pd.DataFrame): Input dataset.
        features (array-like): List of desired features to retain and process.
        simply_clustered (bool, optional): Whether to apply simple clustering to certain features. Defaults to False.
        scaling (bool, optional): Whether to apply scaling to the features. Defaults to False.
        trained_scaler (StandardScaler, optional): Pre-trained scaler for scaling features. If provided, it will be used for scaling. Defaults to None.
        print_weights (bool): If set to true print the mean and standard deviations from the scaler.
    Returns:
        pd.DataFrame: Processed dataframe.
        StandardScaler or None: Scaler used for scaling the features. Returns None if scaling is not applied.
    """
    # Ensure all elements in features are present in df0
    assert set(features).issubset(set(df0.columns))
    df = df0.copy()

    # Retain only desired features
    df = df[features].copy()

    # Drop rows with missing values
    df = df.dropna()

    # Determine feature columns excluding 'Thesis Default' if present
    features = df.drop(columns=['Thesis Default']).columns if 'Thesis Default' in df.columns else df.columns
    
    for feature in features:

        if feature == 'EBITDA (Initial)':
            if simply_clustered:                
                df["EBITDA (Initial)"] = df["EBITDA (Initial)"].apply(ebitda_cluster)
            else:
                df['EBITDA (Initial)'] = df['EBITDA (Initial)'].apply(lambda x: np.log(x) if x >= 1e6 else float('nan'))

        elif feature == 'EV Multiple (Initial)':
            df = df[(df['EV Multiple (Initial)'] > 1) & (df['EV Multiple (Initial)'] < 30)].copy()
            if simply_clustered:
                df[feature] = df[feature].apply(EV_Multiple_cluster)            

        elif feature == 'IC Combined (Initial)':
            df = df[(df[feature] > 0) & (df[feature] < 10)].copy()
            if simply_clustered:
                df[feature] = df[feature].apply(IC_Cluster)

        elif feature == 'LTV (Initial)':
            df = df[(df[feature] > 0) & (df[feature] < 1)].copy()
            if simply_clustered:
                df[feature] = df[feature].apply(LTV_cluster)

        elif feature == 'Security':
            assert(set(df[feature].unique()).issubset(set(['First Lien or Unitranche', 'Second Lien or Mezzanine'])))
            df[feature] = df[feature].apply(Security_cluster)
        
        elif feature == 'Total Net Leverage (Initial)':
            df = df[(df[feature] > 2) & (df[feature] < 10)].copy()
            if simply_clustered:
                df[feature] = df[feature].apply(cluster_net_leverage)

    df = df.dropna()

    if trained_scaler:
        # Extract the target variable
        if 'Thesis Default' in df.columns:
            target = df['Thesis Default']

        # Extract feature columns
        feature_columns = [col for col in df.columns if col != 'Thesis Default']
        feature_columns.sort()
        features = df[feature_columns]

        # Rescale the features using StandardScaler
        scaler = trained_scaler
        scaled_features = scaler.transform(features)

        # Create a dataframe with scaled features
        scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)

        # Concatenate the scaled features with the target variable
        if 'Thesis Default' in df.columns: 
            scaled_df['Thesis Default'] = target.values

        # Rearrange the columns to match the original order
        df = scaled_df[df.columns]
        return df, trained_scaler
        
    elif scaling:
        # Extract the target variable
        if 'Thesis Default' in df.columns:
            target = df['Thesis Default']

        # Extract feature columns
        feature_columns = [col for col in df.columns if col != 'Thesis Default']
        features = df[feature_columns]

        # Rescale the features using StandardScaler
        # scaler = scaler
        scaled_features = scaler.fit_transform(features)

        # Create a dataframe with scaled features
        scaled_df = pd.DataFrame(scaled_features, columns=feature_columns)

        # TODO: Extract means and Standard Deviations from scaler (scaler is StandardScaler form sklearn)
        if print_weights:
            # Extract means and standard deviations from the scaler
            means = scaler.mean_
            std_devs = scaler.scale_

            # Output the means and standard deviations
            means_df = pd.DataFrame(means, index=feature_columns, columns=['Mean'])
            std_devs_df = pd.DataFrame(std_devs, index=feature_columns, columns=['Standard Deviation'])

            print("Means:\n", means_df)
            print("Standard Deviations:\n", std_devs_df)


        # Concatenate the scaled features with the target variable
        if 'Thesis Default' in df.columns: 
            scaled_df['Thesis Default'] = target.values

        # Rearrange the columns to match the original order
        df = scaled_df[df.columns]
        return df, scaler
    
    else:
        return df, None   

def hessian_log_posterior(weights, X, y, sigma_squared):
    """
    Calculate the Hessian of the log-posterior (negative log-posterior) for bayesian logistic regression.

    Args:
        weights (ndarray): The model weights.
        X (ndarray): The feature matrix.
        y (ndarray): The target boolean vector.
        sigma_squared (float): Variance of prior

    Returns:
        -H_log_posterior (ndarray): Lambda for Laplace approximation
    """
    p = 1 / (1 + np.exp(-(2*y-1)*(X @ weights)))
    W = np.diag(p * (1 - p))
    H_log_likelihood = -X.T @ W @ X
    H_log_prior = -(1/sigma_squared) * np.eye(len(weights))
    H_log_posterior = H_log_likelihood + H_log_prior
    
    return -H_log_posterior  

def fit_bayesian_logistic_regression(X, y, sigma_squared=1/2):
    """
    Fit Bayesian Logistic Regression using Laplace approximation.

    Args:
        X (ndarray): The feature matrix.
        y (ndarray): The target vector.
        sigma_squared (float): Covariance paramter of the prior.

    Returns:
        dict: Dictionary containing the MAP estimate, bias and the covariance matrix.
    """
    # Fit logistic regression to get the MAP estimate
    alpha = 1/ (sigma_squared)
    logistic_model = LogisticRegression(penalty='l2', C=1/alpha, solver='saga',)
    logistic_model.fit(X, y)
    
    # Get the MAP estimate of the weights
    w_map = logistic_model.coef_.flatten()

    # Calculate the Hessian of the negative log-posterior at the MAP estimate
    H = hessian_log_posterior(w_map, X, y, 1/alpha)
    
    # Calculate the covariance matrix (inverse of the Hessian)
    cov_matrix = inv(H)
    
    return {
        'w_map': w_map,
        'bias': logistic_model.intercept_,
        'cov_matrix': cov_matrix
    }

def sample_weights(w_map, cov_matrix, num_samples=1000):
    """
    Sample weights from the posterior distribution using the MAP estimate and the covariance matrix.
    
    Args:
        w_map (ndarray): The MAP estimate of the weights.
        cov_matrix (ndarray): The covariance matrix of the weights.
        num_samples (int): The number of samples to draw from the posterior.
        
    Returns:
        samples (ndarray): An array of shape (num_samples, len(w_map)) containing the sampled weights.
    """
    return np.random.multivariate_normal(w_map, cov_matrix, num_samples)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def bayesian_predictive_distribution(X, w_map, bias, cov_matrix, num_samples=1000):
    """
    Compute the predictive distribution for the given input data using Bayesian logistic regression.
    
    Args:
        X (ndarray): The feature matrix for which to compute the predictive distribution.
        w_map (ndarray): The MAP estimate of the weights.
        bias (ndarray): Bias vector of the ridge logistic regression.
        cov_matrix (ndarray): The covariance matrix of the weights.
        num_samples (int): The number of samples to draw from the posterior.
        
    Returns:
        probability_of_default (ndarray): Probaility of Default
    """
    # Sample weights from the posterior distribution
    sampled_weights = sample_weights(w_map, cov_matrix, num_samples)
    
    # Compute the predictions for each set of sampled weights
    predictions = sigmoid(X @ sampled_weights.T + bias)
    
    # Compute the probability of default
    probability_of_default = np.mean(predictions, axis=1)
    
    return probability_of_default

def sample_posterior_BNN(w_map, Lambda_diag):
    """
    Samples weights from the approximate posterior distribution of a Bayesian Neural Network (BNN)
    using a Laplace approximation around the MAP estimate.
    
    Parameters:
    -----------
    w_map : list of torch.Tensor
        The MAP (maximum a posteriori) estimate of the model parameters (weights).
    Lambda_diag : torch.Tensor
        The diagonal elements of the Hessian matrix (Lambda) of the negative log-posterior,
        representing the precision (inverse variance) for each parameter in the approximate Gaussian posterior.
        
    Returns:
    --------
    sampled_params : list of torch.Tensor
        A list of sampled parameters from the posterior distribution, where each sampled parameter
        tensor has the same shape as the corresponding tensor in w_map. These weights can be loaded 
        into the neural network to perform inference under the sampled posterior distribution.
    
    Notes:
    ------
    - This function assumes a diagonal approximation to the Hessian (Lambda), which simplifies the 
      posterior covariance matrix to be diagonal. Each parameter is sampled independently based on
      its variance.
    - The posterior distribution is approximated as a Gaussian: N(w_map, Lambda^-1), where Lambda^-1 
      represents the variances of each weight.
    
    Example Usage:
    --------------
    # Given a trained model with w_map and Lambda_diag computed:
    sampled_weights = sample_posterior_BNN(w_map, Lambda_diag)
    """
    
    sampled_params = []
    # Add a small positive constant to Lambda_diag to prevent division by zero or negative values
    Lambda_diag = torch.clamp(Lambda_diag, min=1e-6).double()
    
    for i, param in enumerate(w_map):
        # Standard deviation for sampling is 1/sqrt(Lambda_diag)
        param = param.double()
        stddev = torch.sqrt(1.0 / Lambda_diag[i])
        
        # Sample from the Gaussian N(w_map, Lambda^-1)
        noise = torch.distributions.Normal(torch.zeros_like(param), stddev).sample()
        sampled_param = param + noise
        sampled_params.append(sampled_param)
    
    return sampled_params

def bayesian_predict(model, X, w_map, Lambda_diag, M=100, device='cpu'):
    """
    Performs Bayesian prediction by averaging the outputs of the ANN over M samples
    from the approximate posterior distribution.
    
    Parameters:
    -----------
    model : nn.Module
        The neural network model to be used for prediction.
    X : np.ndarray
        The input data for which predictions are to be made. Each row should be a feature vector.
    w_map : list of torch.Tensor
        The MAP (maximum a posteriori) estimate of the model parameters (weights).
    Lambda_diag : torch.Tensor
        The diagonal elements of the Hessian matrix (Lambda) of the negative log-posterior,
        representing the precision (inverse variance) for each parameter in the approximate Gaussian posterior.
    M : int, optional (default=100)
        The number of weight samples to draw from the posterior for averaging predictions.
    device : str, optional (default='cpu')
        The device on which to perform predictions ('cpu' or 'cuda').
        
    Returns:
    --------
    np.ndarray
        A 1D NumPy array containing the averaged predictions for each input row in X.
    """
    model.eval()

    # Convert X from NumPy array to PyTorch tensor and move to the specified device
    X_tensor = torch.tensor(X, dtype=torch.float64).to(device)
    predictions = []
    # Ensure w_map and Lambda_diag are in float64 for compatibility with model parameters
    w_map = [param.double() for param in w_map]
    Lambda_diag = Lambda_diag.double()

    for _ in range(M):
        # Sample weights from the posterior
        sampled_weights = sample_posterior_BNN(w_map, Lambda_diag)
        
        # Load the sampled weights into the model
        with torch.no_grad():
            for param, sampled_param in zip(model.parameters(), sampled_weights):
                param.copy_(sampled_param)

        # Make predictions with the sampled weights
        with torch.no_grad():
            outputs = model(X_tensor).squeeze()  # Ensure outputs are 1D for each row in X
            predictions.append(outputs)

    # Stack the predictions and compute the mean along the sample dimension
    predictions = torch.stack(predictions).mean(dim=0)

    # Convert to a 1D NumPy array
    return predictions.cpu().numpy()

class Feature_Model():

    def __init__(self, df: pd.DataFrame, clustering_method='continuous', classifier='logistic_regression', device = 'cpu', penalty='l2', alpha = 3,
                 features = ['FCC (Initial)', 'LTV (Initial)', 'Thesis Default',], second_order_features=['Total Net Leverage (Initial)' ] ,
                   third_order_features=['Total Net Leverage (Initial)'], num_of_epochs=100, k=10, simply_clustered=False, optimizer='saga', mixing_terms=True, l1_ratio=0, hypertune=False, scoring='roc_auc', smote=False, random_state=123, print_weights=False, bnn_laplace='brute-force', drop_out=None) -> None:
        """
        Initialize a feature model specified by its clustering technique and classifier.

        Args:
            df (pd.DataFrame): The labeled dataset.
            clustering_method (str): Specifies the clustering approach; 'continuous' means no clustering.
            classifier (str): The classifier to use.
            device (str): Device to use for training and prediction for the Neural Network approach.
            penalty (str): Penalty type for logistic regression.
            alpha (float): Regularization strength
            features (list): List of features to use, including 'Thesis Default' as a first entry.
            second_order_features (list): List of second-order features.
            third_order_features (list): List of third-order features.
            num_of_epochs (int): Number of epochs for training the neural network classifier.
            k (int): Number of neighbors for KNeighborsClassifier.
            simply_clustered (bool): Whether to use simple clustering.
            optimizer (str): Solver for logistic regression.
            mixing_terms (bool): Parameter for higher order logistic regression deciding whether to use mixed terms for the second order features.
            l1_ratio (float): Elastic Net mixing parameter. Between 0 and 1 where a value of 1 corresponds to Lasso and 0 to Ridge regularization.
            hypertune (bool): Whether to apply hypertuning or not
            scoring (str): String determining the score one uses for grid search. Defaults to AUC. 
            smote (bool): Bool determining whether smote resampling is applied for fitting the model 
            random_state (int): Random state for SMOTE resampling 
            print_weights (bool): Print weights and bias for Logistic Regression
            bnn_laplace(str): String specifiyng the calculation of the hessian in laplace BNN prediction; either 'brute_force' or 'daxberger'

        Returns:
            None
        """
        # make sure the label is included 
        assert 'Thesis Default' in features, "The label 'Thesis Default' must be included in the features."

        self.df = df.sort_index(axis=1)
        self.df_preclustering = df.sort_index(axis=1)
        self.features = list(sorted(features))
        self.simply_clustered = simply_clustered
        self.clustering_method = clustering_method
        self.clustered = False
        self.trained = False
        self.penalty = penalty
        self.optimizer = optimizer
        self.l1_ratio = l1_ratio
        self.bnn_laplace = bnn_laplace
        self.drop_out = drop_out
        DEFAULT_DICT = {
            'EBITDA (Initial)': 35.4 * 1e6, 
            'LTV (Initial)': 0.485, 
            'EV Multiple (Initial)': 8,
            'Total Net Leverage (Initial)': 4.5,
            'IC Combined (Initial)': 2.8,
            'Security': 'First Lien or Unitranche',
            }
        self.default_dict = {key: DEFAULT_DICT[key] for key in self.features if key != 'Thesis Default'}
        self.classifier_method = classifier
        num_of_features_continuous = self.df.drop(columns=['Thesis Default']).shape[1]
        self.classifier, self.sklearn_compatible = { 
            'logistic_regression_higher_order': (LogisticRegression(penalty=penalty, C=1/alpha, solver=optimizer, l1_ratio=self.l1_ratio), True),  
                           'mean': (np.mean, False), 
                           'nn_classifier': (NN_Classifier(input_dim=num_of_features_continuous, hidden_dim=10, dropout_rate=self.drop_out).to(device=device), False),
                           'bayesian_logistic_regression': (LogisticRegression(penalty='l2', C=1/alpha, solver=optimizer), False),
                           'SGD_classifier_higher_order': (SGDClassifier(loss='log_loss', penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,), True),
                           'bnn_classifier': (NN_Classifier(input_dim=num_of_features_continuous, hidden_dim=10, dropout_rate=None).to(device=device), False), 
                           }[classifier]
        self.second_order_list = second_order_features
        self.third_order_list = third_order_features
        self.mixing_terms = mixing_terms
        self.C = 1/alpha
        self.penalty = penalty
        self.hypertune = hypertune
        self.scoring = scoring
        self.smote = smote
        self.random_state = random_state
        self.print_weights = print_weights
        
        # NN parameters
        self.device = device
        self.num_of_epochs = num_of_epochs

        # prepare directories
        subfolder = self.classifier_method
        if self.clustering_method != 'continuous':
            subfolder += '_' + self.clustering_method
        if not os.path.exists(subfolder):
            os.makedirs(subfolder)
        
        # incorporate naming for simple clustering
        self.subfolder = subfolder  + " (clustered)" if self.simply_clustered else subfolder

        feature_list = list( set( [feature[:3] + '_' for feature in self.df.drop(columns='Thesis Default',).columns if feature != 'Security'] ) )
        feature_list = feature_list + ['Secu_'] if 'Security' in self.df.drop(columns='Thesis Default',).columns else feature_list
        feature_list.sort()
        if clustering_method == 'Thesis Clustering':
            feature_list = ['IC_LTV_']
        self.feature_selection = "".join(feature_list)

        feature_subfolder = os.path.join(self.subfolder, self.feature_selection)
        if not os.path.exists(feature_subfolder):
            os.makedirs(feature_subfolder)
        self.feature_subfolder = feature_subfolder
    
    def data_clustering(self, df_test=None):
        """
        Cluster the data for the benchmark model.

        Args:
            df_test (pd.DataFrame): The input dataframe to be clustered.

        Returns:
        pd.DataFrame: Clustered DataFrame for mean prediction or the original DataFrame for continuous method.
        """
        if df_test is None:
            raise ValueError("The input dataframe 'df_test' cannot be None.")

        method = self.clustering_method
        df = df_test.copy()

        if method == 'Thesis Clustering':
            if not set(["LTV (Initial)", "IC Combined (Initial)"]).issubset(set(df.columns)):
                raise ValueError('Error: Required columns "LTV (Initial)" and "IC Combined (Initial)" are missing.')

            if not self.trained:
                self.percentages = np.linspace(33.333, 100, 2, endpoint=False)
                self.ltv_percentiles = [np.percentile(df['LTV (Initial)'], perc) for perc in self.percentages]
                self.ic_percentiles = [np.percentile(df['IC Combined (Initial)'], perc) for perc in self.percentages] 
            ltv_percentiles  = self.ltv_percentiles
            ic_percentiles = self.ic_percentiles

            self.clustered = True
            df['Cluster'] = df.apply(lambda row: benchmark_clustering(ltv_percentiles=ltv_percentiles, ic_percentiles=ic_percentiles, ltv_value=row['LTV (Initial)'], ic_value=row['IC Combined (Initial)']), axis=1)
            assert not df['Cluster'].isnull().values.any()
            return df[['Cluster', 'LTV (Initial)', 'IC Combined (Initial)', 'Thesis Default']].copy()

        elif method == 'continuous':
            return df
        else:
            raise ValueError("Error: Method is not specified. We keep the variables continuous.")
        
    def fit(self, ):
        """
        Fits the model based on the training data stored in self.df.

        The function clusters the data, processes higher order terms if required, 
        and fits the specified classifier.
        """

         # Cluster the data; original dataframe is stored as self.df_preclustering
        self.df = self.data_clustering(self.df)


        # Process higher order terms if applicable
        if self.classifier_method in  ['logistic_regression_higher_order', 'bayesian_logistic_regression', 'SGD_classifier_higher_order']:
            # Add second order terms to the dataframe
            self.second_order_list = [feature for feature in self.second_order_list if feature in self.df.columns]
            for feature in self.second_order_list:
                higher_order_feature = f"{feature}**2"
                self.df[higher_order_feature] = self.df[feature]**2
            
            # Add mixing terms
            if self.mixing_terms:
                for feature_i, feature_j in combinations(self.second_order_list, 2):
                    mixed_feature = f'{feature_i} * {feature_j}'
                    self.df[mixed_feature] = self.df[feature_i] * self.df[feature_j]

             # Add third order terms to the dataframe
            self.third_order_list = [feature for feature in self.third_order_list if feature in self.df.columns]
            
            for feature in self.third_order_list:
                higher_order_feature = f"{feature}**3"
                self.df[higher_order_feature] = self.df[feature]**3
            
            self.df.sort_index(axis=1, inplace=True)

        # Prepare feature matrix X and target vector y
        X = self.df.drop(columns = ['Thesis Default']).values
        y = self.df['Thesis Default'].values

        self.default_mean = y.mean()

        if self.smote:
            smote = SMOTE(random_state=self.random_state)
            X, y = smote.fit_resample(X, y)
            self.smote_probability_scaler = self.default_mean / y.mean()

        self.X = X
        self.y = y
        df = self.df.copy()


        if self.sklearn_compatible:
            # Fit the classifier
            if self.hypertune:
                if self.classifier_method in  ['logistic_regression_higher_order', 'SGD_classifier_higher_order']:
                    param_grid = {
                                    'l2':  {
                                            # Round 1
                                            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                            # Round 2
                                            # 'C': [1/x for x in [0.1, 0.5, 1, 3, 5, 7, 10]],
                                            },
                                    'l1':  {
                                            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                            },
                                    'elasticnet': {
                                            'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                                            'l1_ratio': [0.0, 0.25, 0.5, 0.75, 1.0],
                                            },
                                }[self.penalty]
                    # Initialize GridSearchCV with cross-validation
                    grid_search = GridSearchCV(
                        estimator=self.classifier, 
                        param_grid=param_grid, 
                        cv=5,  # 5-fold cross-validation
                        scoring=self.scoring,
                          # Metric for evaluation
                    )

                    # Fit the grid search to the data
                    grid_search.fit(X, y)
                    self.classifier = grid_search.best_estimator_  
                    # print(f"Best parameters found: {grid_search.best_params_}")
                    self.optimal_hyper_parameters = grid_search.best_params_
            
            # no hypertuning for the other sklearn models as there are no hyperparameters
            else: 
                self.classifier.fit(X, y)
                # TODO: Print weights and bias
                if self.print_weights:
                    weights = self.classifier.coef_[0]  # Get the coefficients (weights)
                    bias = self.classifier.intercept_[0]  # Get the intercept (bias)

                    # Create a DataFrame for better readability
                    weights_df = pd.DataFrame(weights, index=self.df.drop(columns=['Thesis Default']).columns, columns=['Weight'])
                    bias_df = pd.DataFrame([bias], columns=['Bias'])

                    print("Weights:\n", weights_df)
                    print("Bias:\n", bias_df)

            
            self.trained = True

            # Save the trained model
            path = os.path.join(self.feature_subfolder, f"{self.feature_selection}.pkl")
            joblib.dump(self.classifier, path)

        elif self.classifier == np.mean:
            if not self.clustered:
                # Essentially assumes that all loans have the same probability of default and does not use any information from the features
                self.mean_predictor = np.mean(y)
                self.trained = True

                file_path = os.path.join(self.feature_subfolder, "mean_predictor.json")

                # Writing mean predictor to file
                with open(file_path, 'w') as json_file:
                    json.dump(self.mean_predictor, json_file)
            
            elif self.clustering_method == 'Thesis Clustering':
                # Mean predictor by Thesis Cluster
                self.mean_cluster_predictor = {cluster: df[df['Cluster'] == cluster]['Thesis Default'].mean() for cluster in df['Cluster'].unique()}

                # Writing dictionary to file
                file_path = os.path.join(self.feature_subfolder, "mean_predictor.json")
                with open(file_path, 'w') as json_file:
                    json.dump(self.mean_cluster_predictor, json_file)

                self.trained = True

        elif self.classifier_method in ['nn_classifier', 'bnn_classifier',]:
            
            
            # Define L2 (weight_decay) values to hypertune if using cross-validation
            l2_reg_values = [1e-2, 1e-1, 1.0, 1e2, 1e3, ] if self.hypertune else [0]

            if self.hypertune:
                
                best_weight_decay = None
                best_val_auc = 0  # Initialize best AUC as 0 since we're maximizing AUC

                # Initialize KFold cross-validation
                kfold = KFold(n_splits=3, shuffle=True, random_state=self.random_state)

                # Iterate over each L2 regularization strength
                for l2_value in l2_reg_values:
                    avg_val_auc_across_folds = 0.0

                    # K-fold cross-validation loop
                    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
                        print(f"Fold {fold + 1}/{kfold.n_splits} with L2={l2_value}")
                        
                        # Split dataset into training and validation sets for this fold
                        X_train, X_val = X[train_idx], X[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]

                        # Construct dataloaders for this fold
                        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float64), torch.tensor(y_train, dtype=torch.float64))
                        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float64), torch.tensor(y_val, dtype=torch.float64))
                        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

                        # Reinitialize model and optimizer for each fold
                        self.classifier.apply(self.classifier.initialize_weights)
                        self.optimizer = Adam(self.classifier.parameters(), lr=0.001, weight_decay=l2_value)  # Set weight_decay

                        criterion = nn.BCELoss()

                        # Training and Validation Loop for each fold
                        num_epochs = self.num_of_epochs
                        for epoch in range(num_epochs):
                            # Train phase
                            self.classifier.train()
                            running_loss = 0.0
                            for inputs, labels in train_loader:
                                inputs, labels = inputs.to(self.device), labels.to(self.device)

                                self.optimizer.zero_grad()
                                outputs = self.classifier(inputs)
                                outputs = outputs.view(-1)  # Ensure outputs have shape [batch_size]
                                loss = criterion(outputs, labels)  # Labels should already have shape [batch_size]
                                loss.backward()
                                self.optimizer.step()
                                running_loss += loss.item() * inputs.size(0)

                        # Validation phase - Calculate AUC for this fold
                        self.classifier.eval()
                        y_val_true = []
                        y_val_prob = []
                        with torch.no_grad():
                            for inputs, labels in val_loader:
                                inputs, labels = inputs.to(self.device), labels.to(self.device)
                                
                                # Forward pass
                                outputs = self.classifier(inputs)
                                
                                # Ensure output is of shape [batch_size] to match labels
                                outputs = outputs.view(-1)  # Flatten to [batch_size] to match labels
                                
                                # Extend true labels and predicted probabilities
                                y_val_true.extend(labels.cpu().numpy())  # True labels
                                y_val_prob.extend(outputs.cpu().numpy())  # Predicted probabilities

                        # Calculate AUC for this fold
                        fold_auc = roc_auc_score(y_val_true, y_val_prob)
                        print(f"Fold {fold + 1} AUC: {fold_auc:.5f}")
                        avg_val_auc_across_folds += fold_auc

                    # Calculate mean AUC across folds for the current L2 value
                    avg_val_auc_across_folds /= kfold.n_splits
                    print(f"Average AUC for L2={l2_value}: {avg_val_auc_across_folds:.5f}")

                    # Track the best weight_decay value based on AUC
                    if avg_val_auc_across_folds > best_val_auc:
                        best_val_auc = avg_val_auc_across_folds
                        best_weight_decay = l2_value

                # Print the best L2 value found
                print(f"Best L2 regularization (weight_decay): {best_weight_decay} with AUC: {best_val_auc:.5f}")

                # Set the best weight_decay for final training
                weight_decay = best_weight_decay
                self.optimal_hyper_parameters  = {'C': 1/best_weight_decay}  # Used to keep track of best parameters
            else:
                weight_decay = 1/self.C if (self.penalty == "l2") and (1/self.C > 0) else 0
            
            # Construct dataloaders for training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1/7,)

            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float64, requires_grad=True), torch.tensor(y_train, dtype=torch.float64, requires_grad=True))
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float64), torch.tensor(y_val, dtype=torch.float64))
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

            # Define optimizer
            self.optimizer = Adam(self.classifier.parameters(), lr=0.001, weight_decay=weight_decay)  

            criterion = nn.BCELoss()  

            best_val_loss = float('inf')
            train_losses = []
            val_losses = []

            num_epochs = self.num_of_epochs
            
            for epoch in range(num_epochs):

                self.classifier.train()  # Set self.classifier to training mode
                running_loss = 0.0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    self.optimizer.zero_grad()  # Zero the parameter gradients
                    
                    outputs = self.classifier(inputs)  # Forward pass
                    loss = criterion(outputs.squeeze(), labels)  # Calculate loss
                    loss.backward()  # Backward pass
                    self.optimizer.step()  # Update weights
                    
                    running_loss += loss.item() * inputs.size(0)

                avg_train_loss = running_loss / len(train_loader.dataset)
                
                self.classifier.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for X_val, y_val in val_loader:
                        X_val , y_val = X_val.to(self.device), y_val.to(self.device)
                        y_val_pred = self.classifier(X_val)
                        loss = criterion(y_val_pred.squeeze(), y_val)
                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_loader)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    path = os.path.join(self.feature_subfolder, f'best_nn_model_epochs{num_epochs}.pth')
                    torch.save(self.classifier.state_dict(), path)

                print(f"Epoch [{epoch+1}/{num_epochs}], Train loss: {avg_train_loss:.5f}, Val Loss: {avg_val_loss:.5f}")
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)

            plt.figure(figsize=(12, 8))
            plt.semilogy(val_losses, label="Validation Loss")
            plt.semilogy(train_losses, label="Training Loss")
            plt.title("Training vs. Validation Loss")
            plt.xlabel("Number of Epochs")
            plt.ylabel("Binary Classification Error")
            plt.legend()
            plt.grid(True)
            file = os.path.join(self.feature_subfolder, "Val_Train_plot.pdf")
            plt.savefig(file)
            plt.close()

            # BNN Section
            if self.classifier_method == 'bnn_classifier':

                if self.bnn_laplace == 'daxberger':
                    self.la = Laplace(self.classifier, "classification", subset_of_weights='all', hessian_structure="diag")
                    self.la.fit(train_loader=train_loader)

                    self.la.optimize_prior_precision(method='CV', val_loader=val_loader)


                # BRUTE FORCE AUTODIFFERENTIATION 
                else:
                    # Store w_MAP after training
                    w_map = []
                    for param in self.classifier.parameters():
                        w_map.append(param.detach().clone())  # Store the trained parameters as w_MAP

                    self.w_map = w_map  # Save w_MAP for later use

                    # Set model to evaluation mode for BNN posterior calculation
                    self.classifier.eval()

                    # Calculate the diagonal approximation of Hessian Lambda with prior term
                    Lambda_diag = []
                    
                    # Loop over each parameter to compute gradients
                    for param in self.classifier.parameters():
                        if param.requires_grad:
                            # Recompute the loss to get a fresh computation graph
                            outputs = self.classifier(torch.tensor(X_train, dtype=torch.float64).to(self.device))
                            loss = criterion(outputs.squeeze(), torch.tensor(y_train, dtype=torch.float64).to(self.device))
                            
                            # Get gradients with respect to `w_MAP`
                            param_grad = torch.autograd.grad(loss, param, create_graph=True, retain_graph=True)[0]
                            
                            # Compute second derivative for each element
                            diag_hessian = torch.autograd.grad(param_grad, param, grad_outputs=torch.ones_like(param_grad), retain_graph=True)[0]
                            
                            # Add the prior term 1/sigma^2 to each diagonal element
                            prior_term = (weight_decay) * torch.ones_like(diag_hessian)
                            diag_hessian += prior_term
                            
                            Lambda_diag.append(diag_hessian.view(-1))

                    # Flatten the list into a single tensor for Lambda diagonal approximation
                    Lambda_diag = torch.cat(Lambda_diag)

                    # Store Lambda_diag for posterior approximation
                    self.Lambda_diag = Lambda_diag
            self.trained = True

        elif self.classifier_method == 'bayesian_logistic_regression':
            self.blr_param = fit_bayesian_logistic_regression(X,y,sigma_squared=self.C)
            # print(self.blr_param)
            self.trained = True

        else:
            raise ValueError("Error: Classifier method is not specified or invalid.")

    def predict_proba(self, df):
        """
        Predicts the probability of defaults after training the model on the training data.

        Parameters:
            df (pd.DataFrame): The input dataframe for which the probabilities need to be predicted.
            
        Returns:
            np.array: An array of predicted probabilities.
        """
    
        assert self.trained, "The model must be trained before prediction."

        df_test = df.copy()
        df_test['Thesis Default'] = 0  # Needed for API
        df_test = df_test.sort_index(axis=1)
        
        # Cluster the data
        df_test = self.data_clustering(df_test)

        # Process higher order terms if applicable
        if self.classifier_method in [ 'logistic_regression_higher_order', 'bayesian_logistic_regression', 'SGD_classifier_higher_order',]:
            for feature in self.second_order_list:           
                higher_order_feature = f"{feature}**2"
                df_test[higher_order_feature] = df_test[feature]**2
        
            # Add mixing terms
            if self.mixing_terms:
                for feature_i, feature_j in combinations(self.second_order_list, 2):
                    mixed_feature = f'{feature_i} * {feature_j}'
                    df_test[mixed_feature] = df_test[feature_i] * df_test[feature_j]
                
            for feature in self.third_order_list:
                higher_order_feature = f"{feature}**3"
                df_test[higher_order_feature] = df_test[feature]**3
            
            df_test.sort_index(axis=1, inplace=True)

        X_test = df_test.drop(columns=['Thesis Default']).values
        
        # Predicts probabilities using a sklearn-compatible classifier
        if self.sklearn_compatible:
            path = os.path.join(self.feature_subfolder, f"{self.feature_selection}.pkl")
            classifier = joblib.load(path)
            out = classifier.predict_proba(X_test)[:, 1]
            if self.smote:
                out *= self.smote_probability_scaler
            return out 
        
        elif self.classifier == np.mean:
            if not self.clustered:
                # Essentially assumes that all of the loans have the same probability of default and does not use any information from the features
                out = np.ones((X_test.shape[0],)) * self.mean_predictor 
                if self.smote:
                    out *= self.smote_probability_scaler
                return out                
            
            elif self.clustering_method == 'Thesis Clustering':
                assert set(['Thesis Default', 'Cluster']).issubset(set(df_test.columns))
                cluster_mean = self.mean_cluster_predictor # maps cluster name to its cluster mean prediction
                out = df_test['Cluster'].map(cluster_mean).values
                if self.smote:
                    out *= self.smote_probability_scaler
                return out 
        
        elif self.classifier_method == 'bayesian_logistic_regression':
            w_map, cov, bias = self.blr_param['w_map'], self.blr_param['cov_matrix'], self.blr_param['bias']
            out = bayesian_predictive_distribution(X_test, w_map=w_map, bias=bias,cov_matrix=cov, num_samples=1000)
            if self.smote:
                out *= self.smote_probability_scaler
            return out
        
        elif self.classifier_method == 'bnn_classifier':

            out = 0
            
            if self.bnn_laplace == 'daxberger':
                X_test = torch.tensor(X_test)
                out = self.la(X_test, link_approx='mc').numpy()

            else:
                # Perform Bayesian prediction
                out = bayesian_predict(
                    model=self.classifier,
                    X=X_test,
                    w_map=self.w_map,
                    Lambda_diag=self.Lambda_diag,
                    M=1000,
                    device=self.device  # Use the model's device
                )

            if self.smote:
                out *= self.smote_probability_scaler
            return out

        # Predicts probabilities using a neural network classifier
        elif self.classifier_method == 'nn_classifier':
            X_test = torch.tensor(X_test)
            path = os.path.join(self.feature_subfolder, f'best_nn_model_epochs{self.num_of_epochs}.pth')
            state_dict = torch.load(path)
            self.classifier.load_state_dict(state_dict=state_dict)

            self.classifier.eval()

            with torch.no_grad():
                
                X_test = X_test.to(self.device)
                y_test_pred = self.classifier(X_test)
            
            out = y_test_pred.detach().to(self.device).numpy()
            if self.smote:
                out *= self.smote_probability_scaler
            return out 

        else:
            raise ValueError("Error: No valid classifier method specified.")
        
        return None    

    def plot_auc(self, y_true, y_score):
        """
        Plots the Receiver Operating Characteristic (ROC) curve and highlights the point where the 
        False Positive Rate (FPR) is closest to 0.2.

        Parameters:
            y_true (array-like): True binary labels.
            y_score (array-like): Target scores, probability estimates of the positive class.
        """
        
        # Ensure inputs are valid
        assert len(y_true) == len(y_score), "y_true and y_score must have the same length."
        assert len(y_true) > 0, "y_true and y_score must not be empty."

        # Compute ROC curve and ROC area
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # Find the index where FPR is closest to 0.2
        idx_fpr_0_2 = np.argmin(np.abs(fpr - 0.2))
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.scatter(fpr[idx_fpr_0_2], tpr[idx_fpr_0_2], color='red', s=100, )
        
        # Annotate the point with FPR, TPR, and threshold values
        plt.text(
            fpr[idx_fpr_0_2] + 0.08, tpr[idx_fpr_0_2] - 0.1,
                f'FPR={fpr[idx_fpr_0_2]:.2f}\nTPR={tpr[idx_fpr_0_2]:.2f}\nThreshold={thresholds[idx_fpr_0_2]:.2f}',
                fontsize=12, ha='center', color='red',
        )
        
        # Plot labels and title
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")

        # Save and close the plot 
        plt.savefig(os.path.join(self.feature_subfolder, 'Receiver Operating Characteristic.png'))
        plt.close()

    def evaluation_metrics(self, y_true, y_score, thresholds=np.linspace(0.05, 0.2, 4, endpoint=True, ), fpr_range=(0.0, 0.4), ):
        """
        Calculate various evaluation metrics for binary classification models, including precision, recall, F1 score, AUC, Brier score, and Expected Calibration Error (ECE).
        The function also evaluates metrics at multiple threshold values, providing insights into the model's performance under different decision boundaries.
        
        Args:
            y_true (array-like): True binary labels (0s and 1s).
            y_score (array-like): Predicted scores (probabilities or confidence scores).
            thresholds (array-like, optional): Array of threshold values for converting scores to binary predictions. 
                Default is np.linspace(0.05, 0.2, 4, endpoint=True).
            fpr_range (tuple, optional): Range of False Positive Rate (FPR) for restricted AUC calculation. Default is (0.0, 0.4).
        Returns:
            list: A list containing overall metrics followed by metrics for each threshold. The list structure is:
                [roc_auc, auc_restricted, log_loss, mean_squared_error, precision_t1, recall_t1, 1-specificity_t1, f0.5_t1, f1_t1, f2_t1, ..., precision_tN, recall_tN, 1-specificity_tN, f0.5_tN, f1_tN, f2_tN]
        """
        out = []
        
        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Calculate AUC for the specified FPR range
        fpr_indices = np.where((fpr >= fpr_range[0]) & (fpr <= fpr_range[1]))[0]
        fpr_sub = fpr[fpr_indices]
        tpr_sub = tpr[fpr_indices]
        auc_restricted = auc(fpr_sub, tpr_sub) if len(fpr_sub) > 1 else float('nan')

        # Append overall metrics to the output list
        out.extend([roc_auc, auc_restricted, log_loss(y_true=y_true, y_pred=y_score), mean_squared_error(y_true=y_true, y_pred=y_score)])
    
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            
            # Calculate confusion matrix and derived metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
            precision = precision_score(y_true, y_pred) if np.max(y_pred) > 0 else float('nan')
            recall = recall_score(y_true, y_pred)
            beta = 0.5
            f_half = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if beta**2 * precision + recall != 0 else float('nan')
            f1 = f1_score(y_true, y_pred) if (recall * precision) > 0 else float('nan')
            beta = 2
            f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall) if beta**2 * precision + recall != 0 else float('nan')
            
            # Append metrics for the current threshold to the output list
            out.extend([precision, recall, 1-specificity, f_half, f1, f2])
        
        return out

    def plot_feature_pd(self, feature=['LTV (Initial)'], trained_scaler=None, scaling=False, convert_to_lossrate=False, recovery_rate=0.6, sub_delta=0.15):
        """
        Plot how the probability of default changes with respect to a specific feature while fixing all other features.

        Args:
            feature (list): List containing one feature name to plot against the probability of default.
            trained_scaler (object, optional): Trained scaler object for preprocessing. Default is None.
            scaling (bool, optional): Whether to apply scaling during preprocessing. Default is False.
            convert_to_lossrate (bool, optional): Whether to convert probability of default to annualized expected loss rate. Default is False.
            recovery_rate (float, optional): Recovery rate used for converting to loss rate. Default is 0.5.
            sub_delta (float, optional): Difference in recovery rate for subordinated debt

        Returns:
            None: Saves and displays the plot of probability of default or loss rate against the specified feature.
        """
        assert set(feature).issubset(set(self.features)) and len(feature) == 1, f"Invalid feature: {feature}"

        num_points = 200
        domains = {
            "LTV (Initial)": np.linspace(0.01, 0.99, num_points),
            "EBITDA (Initial)": np.linspace(1e6 + 1, 300 * 1e6, num_points * 30),
            "EV Multiple (Initial)": np.linspace(1.01, 29.9, num_points),
            "IC Combined (Initial)": np.linspace(0.01, 9.99, num_points),
            "Total Net Leverage (Initial)": np.linspace(2.1, 7.5, num_points),
            "Security": ['First Lien or Unitranche', 'Second Lien or Mezzanine']
        }

        # Ensure input feature is valid and retrieve x values
        x_values = domains[feature[0]]

        d = self.default_dict
        fixed_features = [f for f in self.features if f not in feature and f != 'Thesis Default']
        fixed_features.sort()

        # Create a DataFrame to store the grid data
        df_surface = pd.DataFrame({
            feature[0]: np.array(x_values),
        })

        info_str = ''
        for key in fixed_features:
            df_surface[key] = d[key]
            info_str += f"{key}={d[key]}, "

        df_surface = df_surface[list(sorted(df_surface.columns))]

        # Preprocess data if classifier method is not 'mean'
        if self.classifier_method != 'mean':
            df_surface, _ = pre_processing(df_surface, list(df_surface.columns), simply_clustered=self.simply_clustered, trained_scaler=trained_scaler, scaling=scaling)

        # Predict probability of default
        prob_of_default = self.predict_proba(df_surface)

        # Convert to loss rate if specified
        if convert_to_lossrate:
            if feature[0] == 'Security':
                y = (1 + (1 - np.array([recovery_rate, recovery_rate-sub_delta])).reshape(2, 1) * prob_of_default) ** (1 / 3) - 1
            else:
                y = (1 + (1 - recovery_rate) * prob_of_default) ** (1 / 3) - 1
            
            title_plot = 'Annualized Expected Loss Rate'
        else:
            y = prob_of_default
            title_plot = 'Probability of Default'

        # Plotting
        plt.figure(figsize=(12, 8))
        if feature[0] == 'Security':
            out = y.reshape(2,)
            plt.bar(x_values, list(out))
        else: 
            plt.plot(np.array(x_values), y)

        xlabel = feature[0].replace(" (Initial)", "").replace(" Combined","")
        plt.xlabel(xlabel)

        # Format y-axis as percentage with 2 decimal places
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2%}'))

        plt.ylabel(title_plot)

        # Split title into two lines
        title = f'{title_plot}{" with Floors and Caps" if self.simply_clustered else " with " + self.clustering_method + "Clustering"} \nusing {self.classifier_method.replace("_", " ").title()} Model'
        plt.title(title)

        # Remove gridlines
        plt.grid(False)

        plt.savefig(os.path.join(self.feature_subfolder, f'{title_plot.replace(" ", "_")}_{feature[0].replace(" ", "_")}_Plot.pdf'), format='pdf')
        plt.close()



