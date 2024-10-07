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

import torch 
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, SGD, SparseAdam
import joblib

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, roc_curve, auc, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, brier_score_loss, log_loss
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, Matern, DotProduct, RationalQuadratic
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier

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
        out = torch.sigmoid(out)

        if self.dropoutrate is not None:
            out = self.dropout1(out)

        out = self.output_layer(out)
        out = torch.sigmoid(out)

        return out


class BNN_Classifier(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=3, output_dim=1, dropout_rate=None, sigma_prior=1.0):
        super(BNN_Classifier, self).__init__()
        self.hidden_layer1 = nn.Linear(input_dim, hidden_dim, dtype=torch.float64)
        self.dropoutrate = dropout_rate
        self.sigma_prior = sigma_prior

        if dropout_rate is not None:
            if not (0 <= dropout_rate <= 1):
                raise ValueError("Dropout rate must be between 0 and 1")
            self.dropout1 = nn.Dropout(p=dropout_rate)

        self.output_layer = nn.Linear(hidden_dim, output_dim, dtype=torch.float64)
        self.apply(self.initialize_weights)

    def initialize_weights(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, input):
        out = self.hidden_layer1(input)
        out = torch.sigmoid(out)

        if self.dropoutrate is not None:
            out = self.dropout1(out)

        out = self.output_layer(out)
        out = torch.sigmoid(out)

        return out

    def log_prior(self):
        log_prior = 0.0
        for param in self.parameters():
            log_prior += -0.5 * torch.sum(param.pow(2)) / (self.sigma_prior ** 2)
        return log_prior

    def log_likelihood(self, x, y):
        y_pred = self.forward(x)
        likelihood = torch.distributions.Bernoulli(y_pred).log_prob(y)
        return likelihood.sum()

    def log_posterior(self, x, y):
        return self.log_likelihood(x, y) + self.log_prior()

    def hessian(self, x, y):
        loss = -self.log_posterior(x, y)
        loss.backward(create_graph=True)
        hessian = []
        for param in self.parameters():
            grad2 = torch.autograd.grad(loss, param, retain_graph=True, create_graph=True)[0]
            hess = torch.autograd.grad(grad2, param, retain_graph=True)[0]
            hessian.append(hess)
        return hessian

    def predict_monte_carlo(self, x, num_samples=100):
        """
        Monte Carlo approximation for Bayesian prediction.

        Args:
            x (torch.Tensor): Input data.
            num_samples (int): Number of Monte Carlo samples.

        Returns:
            np.ndarray: Mean of predictions across sampled weights as a NumPy array.
        """
        preds = []
        for _ in range(num_samples):
            sampled_params = []
            for param in self.parameters():
                # Sample weight perturbations from a normal distribution
                noise = torch.randn_like(param) * self.sigma_prior
                sampled_param = param + noise
                sampled_params.append(sampled_param)
            
            # Temporarily set model parameters to the sampled ones
            self.set_parameters(sampled_params)
            
            # Compute forward pass and store prediction
            preds.append(self.forward(x))
        
        # Restore original parameters (optional depending on your use case)
        self.apply(self.initialize_weights)
        
        # Convert the mean of the predictions to a NumPy array and return
        return torch.stack(preds).mean(0).detach().numpy()

    def set_parameters(self, params):
        """
        Helper function to set the model parameters with given values.
        
        Args:
            params (List[torch.Tensor]): List of parameters to set.

        Returns:
            None
        """
        with torch.no_grad():
            for p, new_p in zip(self.parameters(), params):
                p.copy_(new_p)    


def train_bnn(model, X_train, y_train, learning_rate=0.01, num_epochs=1000):
    # Define the optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    
    # Loss function (Binary Cross Entropy)
    loss_fn = nn.BCELoss()

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        
        # Forward pass
        y_pred = model(X_train)
        
        # Compute the loss
        loss = loss_fn(y_pred, y_train)
        
        # Add prior to the loss for Bayesian inference (Regularization)
        loss -= model.log_prior()
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def benchmark_clustering(ltv_percentiles, ic_percentiles, ic_value, ltv_value):
    """
    Clusters IC and LTV values based on provided percentiles.

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
    Helper function to categorize EBITDA values into clusters.

    Args:
        x (float): EBITDA value.

    Returns:
        float: Log-transformed EBITDA value for clustering.
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
    Helper function to categorize Loan-to-Value (LTV) ratio values into clusters.

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
    Helper function to categorize Interest Coverage (IC) ratio values into clusters.

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
    Helper function to categorize Enterprise Value (EV) multiples into clusters.

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
    Helper function to categorize net leverage values into clusters.

    Args:
        x (float): Net leverage value.

    Returns:
        int: Cluster index based on the net leverage value.
        float: NaN if the value does not match any predefined categories.
    """
    if x > 0:
        return x
    elif x > 6.5:
        return 6.5
    else:
        return float('nan')

def base_case_cum_net_return(fprs, tprs, thresholds=np.linspace(0.01, 0.99, 99, endpoint=True), 
                             loan_life=2.5, default_rate=0.0476):
    """
    Calculate a metric comparing different models or scenarios based on loan performance.

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
    Num_of_loans = 1000
    overall_default_rate = default_rate
    bad_loans = np.ceil(Num_of_loans * overall_default_rate)
    good_loans = Num_of_loans - bad_loans

    r_per_deal = 0.08
    recovery_rate = 0.5
    r_alternative = 0.055
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

def pre_processing(df0, features, simply_clustered=False, scaling=True, US_EU_Only=True, trained_scaler=None, scaler=StandardScaler(), print_weights=False):
    """
    Processes the input dataframe and returns a cleaned and preprocessed dataframe.

    Args:
        df0 (pd.DataFrame): Input dataset.
        features (array-like): List of desired features to retain and process.
        simply_clustered (bool, optional): Whether to apply simple clustering to certain features. Defaults to False.
        scaling (bool, optional): Whether to apply scaling to the features. Defaults to True.
        US_EU_Only (bool, optional): Whether to retain only US and Europe regions. Defaults to True.
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

        if feature == 'Region':

            if US_EU_Only:
                df = df[df['Region'] != 'Other'].copy()
            
            one_hot_encoder = OneHotEncoder()
            one_hot_encoded = one_hot_encoder.fit_transform(df[['Region']])
            columns = one_hot_encoder.get_feature_names_out(['Region'])
            one_hot_encoded_df = pd.DataFrame(one_hot_encoded.toarray(), columns=columns, index=df.index)

            df = pd.concat([df, one_hot_encoded_df], axis=1)
            drop_reg_col = ['Region', 'Region_Other']
            if US_EU_Only:
                drop_reg_col.append('Region_Europe')
            df = df.drop(columns=[col for col in drop_reg_col if col in df.columns])
            
            if not US_EU_Only:
                for col in ['Region_US', 'Region_Europe']:
                    if col not in df.columns:
                        df[col] = 0

        elif feature == 'EBITDA (Initial)':
            if simply_clustered:                
                df["EBITDA (Initial)"] = df["EBITDA (Initial)"].apply(ebitda_cluster)
            else:
                df['EBITDA (Initial)'] = df['EBITDA (Initial)'].apply(lambda x: np.log(x) if x >= 1e6 else float('nan'))

        elif feature == 'EV Multiple (Initial)':
            df = df[(df['EV Multiple (Initial)'] > 1) & (df['EV Multiple (Initial)'] < 30)].copy()
            if simply_clustered:
                df[feature] = df[feature].apply(EV_Multiple_cluster)            

        elif feature == 'IC (Initial)':
            df = df[(df[feature] > 0) & (df[feature] < 10)].copy()
            if simply_clustered:
                df[feature] = df[feature].apply(IC_Cluster)

        elif feature == 'Ownership':
            df['Ownership'] = df['Ownership'].apply(lambda x: 1 if x == 'Sponsor' else (0 if x == 'Private Corporate' else float('nan')))

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

        # Extract means and Standard Deviations from scaler (scaler is StandardScaler form sklearn)
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
        y (ndarray): The target vector.
        sigma_squared (float): Diagonal entry for prior

    Returns:
        hessian (ndarray): The Hessian matrix of the log-posterior.
    """
    p = 1 / (1 + np.exp(-X @ weights))
    W = np.diag(p * (1 - p))
    H_log_likelihood = -X.T @ W @ X
    H_log_prior = -(1/sigma_squared) * np.eye(len(weights))
    H_log_posterior = H_log_likelihood + H_log_prior
    
    return -H_log_posterior  # We return negative because we minimize in scipy

# TODO: Adapt the optimizer!
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
    alpha = 1/ (2*sigma_squared)
    logistic_model = LogisticRegression(penalty='l2', C=1/alpha, solver='lbfgs',)
    logistic_model.fit(X, y)
    
    # Get the MAP estimate of the weights
    w_map = logistic_model.coef_.flatten()

    # Calculate the Hessian of the negative log-posterior at the MAP estimate
    H = hessian_log_posterior(w_map, X, y, alpha)
    
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


class Feature_Model():

    def __init__(self, df: pd.DataFrame, clustering_method='continuous', classifier='logistic_regression', device = 'cpu', penalty='l2', alpha = 3,
                 features = ['FCC (Initial)', 'LTV (Initial)', 'Thesis Default',], second_order_features=['Total Net Leverage (Initial)' ] ,
                   third_order_features=['Total Net Leverage (Initial)'], num_of_epochs=100, k=10, simply_clustered=False, optimizer='saga', mixing_terms=True, l1_ratio=0.15, hypertune=False, scoring='roc_auc', smote=False, random_state=123, print_weights=False) -> None:
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
        DEFAULT_DICT = {
            'EBITDA (Initial)': 35.4 * 1e6, 
            'LTV (Initial)': 0.485, 
            'Maturity': 5.0,  
            'Region': 'US',  
            'EV Multiple (Initial)': 8,
            'Ownership': 'Sponsor',
            'Total Net Leverage (Initial)': 4.5,
            'IC (Initial)': 3,
            'Security': 'First Lien or Unitranche',
            }
        self.default_dict = {key: DEFAULT_DICT[key] for key in self.features if key != 'Thesis Default'}
        self.classifier_method = classifier
        num_of_features_continuous = self.df.drop(columns=['Thesis Default']).shape[1]
        self.classifier, self.sklearn_compatible = {
            'logistic_regression': (LogisticRegression(penalty=None, solver=optimizer, C=1/alpha), True), 
            'logistic_regression_higher_order': (LogisticRegression(penalty=penalty, C=1/alpha, solver=optimizer), True),
            'regularized_logistic_regression': (LogisticRegressionCV(penalty=penalty, cv=3, solver=optimizer, Cs=[1/x for x in range(1, 10, 2)]), True), 
            'mean': (np.mean, False), 
            'nn_classifier': (NN_Classifier(input_dim=num_of_features_continuous, hidden_dim=5, dropout_rate=0.5).to(device=device), False),
            'bayesian_logistic_regression': (LogisticRegression(penalty='l2', C=1/alpha,), False),
            'SGD_classifier_higher_order': (SGDClassifier(loss='log_loss', penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,), True),
            'SGD_classifier': (SGDClassifier(loss='log_loss', penalty=penalty, alpha=alpha, l1_ratio=l1_ratio,), True),
            'bnn_classifier': (BNN_Classifier(input_dim=num_of_features_continuous, hidden_dim=5, output_dim=1, dropout_rate=None), False), 
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
        Cluster the data using various techniques (self.clustering_method).

        Args:
            df_test (pd.DataFrame): The input dataframe to be clustered.

        Returns:
        pd.DataFrame: One-hot encoded DataFrame for mean prediction, the original DataFrame for continuous method,
                      or a DataFrame with discrete versions of continuous features.
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
        if self.classifier_method in  ['logistic_regression_higher_order', 'regularized_logistic_regression', 'bayesian_logistic_regression', 'SGD_classifier_higher_order']:
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

        if self.smote:
            smote = SMOTE(random_state=self.random_state)
            X, y = smote.fit_resample(X, y)

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
                # Print weights and bias
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

        elif self.classifier_method in ['nn_classifier', 'bnn_classifier']:

            # Construct dataloaders for training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1/7,)

            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float64, requires_grad=True), torch.tensor(y_train, dtype=torch.float64, requires_grad=True))
            val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float64), torch.tensor(y_val, dtype=torch.float64))
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

            # Define optimizer
            self.optimizer = Adam(self.classifier.parameters(), lr=0.001)  

            # TODO: Potentially add weight criteria
            # Define loss function
            criterion = nn.BCELoss()  

            best_val_loss = float('inf')
            train_losses = []
            val_losses = []

            num_epochs = self.num_of_epochs
            for epoch in range(num_epochs):

                if self.classifier_method == 'bnn_classifier':
                    criterion -= self.classifier.log_prior()


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

            plt.figure(figsize=(8, 12))
            plt.semilogy(val_losses, label="Validation Loss")
            plt.semilogy(train_losses, label="Training Loss")
            plt.title("Train vs. Validation Loss")
            plt.xlabel("Number of Epochs")
            plt.ylabel("Binary Classification Error")
            plt.legend()
            plt.grid(True)
            file = os.path.join(self.feature_subfolder, "Val_Train_plot.png")
            plt.savefig(file)
            plt.close()
            self.trained = True

        elif self.classifier_method == 'bayesian_logistic_regression':
            self.blr_param = fit_bayesian_logistic_regression(X,y,sigma_squared=self.C/2)
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

        # TODO: Can add potentially more classifiers
        # Process higher order terms if applicable
        if self.classifier_method in [ 'logistic_regression_higher_order', 'regularized_logistic_regression', 'bayesian_logistic_regression', 'SGD_classifier_higher_order',]:
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
            return classifier.predict_proba(X_test)[:, 1]
        
        elif self.classifier == np.mean:
            if not self.clustered:
                # Essentially assumes that all of the loans have the same probability of default and does not use any information from the features
                return np.ones((X_test.shape[0],)) * self.mean_predictor               
            
            elif self.clustering_method == 'Thesis Clustering':
                assert set(['Thesis Default', 'Cluster']).issubset(set(df_test.columns))
                cluster_mean = self.mean_cluster_predictor # maps cluster name to its cluster mean prediction
                
                return df_test['Cluster'].map(cluster_mean).values

        elif self.classifier_method == 'bayesian_logistic_regression':
            w_map, cov, bias = self.blr_param['w_map'], self.blr_param['cov_matrix'], self.blr_param['bias']
            return bayesian_predictive_distribution(X_test, w_map=w_map, bias=bias,cov_matrix=cov, num_samples=1000)

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
            
            return y_test_pred.detach().to(self.device).numpy()

        else:
            raise ValueError("Error: No valid classifier method specified.")
        
        return None    
  
    def plot_heatmap(self, features=["IC (Initial)", "LTV (Initial)", ], trained_scaler=None, scaling=False,):
        """
        Plot the heat map surface of the predicted probabilities of default with respect to two features where we fix the other features if there are any.

        Parameters:
            features (list): List of two features to plot the heatmap for.
            trained_scaler: Scaler object if scaling is required (default is None).
            scaling (bool): Flag to indicate if scaling is required (default is False).
        """
        assert set(features).issubset(set(self.features)), "Specified features must be a subset of training features."
        assert len(features) == 2, "Exactly two features must be specified."
        
        # Generate grid data using np.linspace
        num_points = 200
        # TODO: Can be updated
        domains = {
            "LTV (Initial)": np.linspace(0.01, 1, num_points),
            "FCC (Initial)": np.linspace(0.01, 5, num_points),
            "EBITDA (Initial)": np.linspace(1e6+1, 250e6, num_points*30),
            "EV Multiple (Initial)": np.linspace(4, 15, num_points),
            "IC Combined (Initial)": np.linspace(0.01, 5, num_points),
            "Ownership": ["Sponsor", 'Private Corporate'],
            "Region": ["Europe", "US"],
            "Total Net Leverage (Initial)": np.linspace(2.5, 7.5, num_points),
            "Security": ['First Lien or Unitranche', 'Second Lien or Mezzanine'],
        }
        
        # Validate input features
        if set(features).issubset(set(domains.keys())):
            d = self.default_dict
            fixed_features = [feature for feature in self.features if not ((feature in features) or (feature == 'Thesis Default'))]
            fixed_features.sort()
        
            x_values = domains[features[0]]
            y_values = domains[features[1]]

            # Create coordinate matrices using np.meshgrid
            x_grid, y_grid = np.meshgrid(x_values, y_values)

            # Create a DataFrame to store the grid data
            df_surface = pd.DataFrame({
                features[0]: x_grid.flatten(),
                features[1]: y_grid.flatten()
            })
            
            info_str = ''
            for key in fixed_features:
                    df_surface[key] = d[key]
                    info_str += (key + '=' + str(d[key]) + ', ')

            if self.classifier_method != 'mean':           
                df_surface, _ = pre_processing(df_surface, list(df_surface.columns), simply_clustered=self.simply_clustered, trained_scaler=trained_scaler, scaling=scaling, )
                
            prob_of_default = self.predict_proba(df_surface)

            # Create a heatmap for test set
            plt.figure(figsize=(20,10))
           
            if "EBITDA (Initial)" == features[0]:
                plt.xscale('log')
            elif "EBITDA (Initial)" == features[1]:
                plt.yscale('log')

            plt.scatter(x_grid.flatten(),  y_grid.flatten(), c=prob_of_default, cmap='coolwarm')
            plt.colorbar(label='Probability of Default')
            plt.xlabel(f'{features[0]}')
            plt.ylabel(f'{features[1]}')
            
            clustering = "simple" if self.simply_clustered else self.clustering_method
            title = f'Heat Map for {clustering} clustering and using {self.classifier_method} classifier for {features[0][:3]} and {features[1][:3]}'
            
            if fixed_features != []:
                title += ' and fixed ' + info_str
                title = title[:-1]

            plt.title(title)
            plt.grid(False)
            plt.savefig(os.path.join(self.feature_subfolder, f'Heatmap {features[0][:3]} and {features[1][:3]}' + '.pdf'), format='pdf')
            plt.close()

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

    def plot_calibration_curve(self, y_true, y_prob, n_bins=30, y_lim=None, x_lim=None):
        """
        Plot a calibration curve to compare predicted probabilities to actual outcomes.

        This function splits the interval [0, 1] into `n_bins` uniformly. For each bin, it computes
        the mean of all entries of `y_prob` which fall into this bin and the corresponding entries 
        of `y_true` to compute the default frequency (average). The plot shows the mean predicted 
        probability per bin (x-axis) against the actual default frequency per bin (y-axis).

        Args:
            y_true (array-like): True binary labels (0 or 1).
            y_prob (array-like): Predicted probabilities for the positive class.
            n_bins (int, optional): Number of bins to use for binning predicted probabilities. Default is 30.
            y_lim (tuple, optional): y-axis limits (min, max). Default is None.
            x_lim (tuple, optional): x-axis limits (min, max). Default is None.

        Returns:
            None: Displays and saves the calibration plot.
        """

        # Ensure input arrays are valid
        if len(y_true) != len(y_prob):
            raise ValueError("Length of y_true and y_prob must be the same.")
        if n_bins <= 0:
            raise ValueError("Number of bins must be a positive integer.")

        # Calculate calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)

        # Plot the calibration curve
        plt.plot(prob_pred, prob_true, marker='o', linestyle='-')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line for perfect calibration

        plt.xlabel('Average Predicted Probability per Bin')
        plt.ylabel('Actual Default Frequency per Bin')

        # Set axis limits
        plt.xlim(left=0, right=x_lim if x_lim else prob_pred.max())
        plt.ylim(bottom=0, top=y_lim if y_lim else prob_true.max() + 0.04)

        plt.title('Calibration Plot')

        # Add count for each bin
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_counts = np.histogram(y_prob, bins=bin_edges)[0]
        for i, count in enumerate(bin_counts):
            if i < len(prob_pred):
                plt.text(prob_pred[i], prob_true[i], f'({count})', fontsize=8, va='center', ha='center')

        plt.grid(True)

        # Set x and y tick labels as percentages
        plt.gca().xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
        plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

        # Save the plot
        plt.savefig(os.path.join(self.feature_subfolder, 'Calibration Plot.png'))
        plt.close()

    def plot_precision_recall_curve(self, y_true, y_score):
        """
        Plot the precision-recall curve based on true labels and predicted scores.

        Args:
            y_true (array-like): True binary labels (0 or 1).
            y_score (array-like): Predicted scores (probabilities or confidence scores).

        Returns:
            None: Saves and displays the precision-recall curve plot.
        """
        # Check input validity
        if len(y_true) != len(y_score):
            raise ValueError("Length of y_true and y_score must be the same.")

        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true, y_score)

        # Plot precision-recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(self.feature_subfolder, 'Precision-Recall Curve.png'))
        plt.close()

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
            "FCC (Initial)": np.linspace(0.01, 6, num_points),
            "EBITDA (Initial)": np.linspace(1e6 + 1, 300 * 1e6, num_points * 30),
            "EV Multiple (Initial)": np.linspace(1.01, 29.9, num_points),
            "IC Combined (Initial)": np.linspace(0.01, 9.99, num_points),
            "Ownership": ["Sponsor", 'Private Corporate'],
            "Region": ["Europe", "US"],
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
