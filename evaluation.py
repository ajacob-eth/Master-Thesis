from Feature_Model import Feature_Model, pre_processing, Security_PreCluster, base_case_cum_net_return, sigmoid, bayesian_predictive_distribution
import pandas as pd
import numpy as np
import xlwings as xw
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from collections import Counter
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy import stats

OLD_THESIS_SET = True
FILE_PATH = 'Your_Dataset.csv'
EVAL_PATH = r"Model Evaluation.xlsm" # path to your evaluation workbook in excel which formats automatically your outputs

# Use a specific dataset to predict probabilities using your model and for Bayesian Logistic Regression get uncertainty interval for your prediciton
SPECIFIC_DATASET = False
DATA_NAME = r'loan_portfolio.xlsx' # path to your loan data 
UNCERTAINTY_INTERVAL = False

# Select features to test out
FEATURES = [
             'EBITDA (Initial)', 
           'LTV (Initial)',  
            'EV Multiple (Initial)', 
            'IC Combined (Initial)',
            'Security',
            'Total Net Leverage (Initial)',
            # 'Region', 
            # 'Ownership',
            ]
SPECIFIC_FEATURE_COMBO = (True, [ 
            'EBITDA (Initial)', 
            'LTV (Initial)',  
            'EV Multiple (Initial)', 
            'IC Combined (Initial)',
            'Security',
             'Total Net Leverage (Initial)', 
            ])
# uncomment where you want to have second/third order terms 
SECOND_ORDER=[
            # 'LTV (Initial)',  
            # 'EV Multiple (Initial)', 
            # 'IC Combined (Initial)',
            #  'Total Net Leverage (Initial)',
            ]
THIRD_ORDER=[
            # 'EBITDA (Initial)', 
            # 'LTV (Initial)',  
            # 'EV Multiple (Initial)', 
            # 'IC Combined (Initial)',
            #  'Total Net Leverage (Initial)',
            # 'FCC (Initial)',
]

# add mixing terms
MIXING_TERMS = False

# Extract the master thesis dataset
GENERATE_FILTERED_DATASET = False

# adds floors and caps
CLUSTERED = False

# set to false when using benchmark model
SCALING = True

CLASSIFIERS = [
            # ('continuous', 'logistic_regression'), 
            # ('continuous', 'logistic_regression_higher_order'),
            ('continuous', 'bayesian_logistic_regression'),
            # ('continuous', 'bnn_classifier'),
            ######### BeMA Model###############
            # ('Thesis Clustering', 'mean'), 
            ###########################
            # ('continuous', 'regularized_logistic_regression'),
            # ('continuous', 'nn_classifier'),
            # ('continuous', 'SGD_classifier'),
            # ('continuous', 'SGD_classifier_higher_order'),
            # ('continuous', 'mean'),
            ]
# includes cross vlaidation for higher order logistic regression model
HYPERTUNE = False
SCORING = 'roc_auc' # 'neg_brier_score'

# specifies regularization and optimizer for standard logistic regression models
REGULARIZATION_STRENGTH = 10
l1_ratio = 0.15
REGULARIZER = 'l2'
OPTIMIZER = 'saga'

# for neural network
NUM_OF_EPOCHS = 50
SMOTE = False
# prints the parameters for scaler and model
PRINT_WEIGHTS = False

# Evaluations
NUM_OF_TRAINING = 100

# Used for weight distribution and specific prediction
TRAIN_SIZE = 1
GET_WEIGHTS = True

# Threshold optimization set for PAM measure
BASE_CASE_THRESHOLDS = np.linspace(0.01, 0.495, 99, endpoint=True)
BASE_CASE_LOAN_LIFE = 3
THRESHOLDS = np.array([0.0481])
AUC_RANGE = (0, 0.4)
COMPUTE_COFUSION_METRICS = True
FEATURE_PLOTS = False
SINGLE_FEATURE_PLOTS = True
CONVERT_TO_LOSSRATE = False
RECOVERY_RATE = 0.6
SUB_DELTA = 0.2

EVALUATION_PLOTS = False
N_BINS = 50
Y_LIM = 0.3
X_LIM = 0.3

def generate_combinations(lst):
    """
    Generate all possible combinations with at least three elements from a given list.

    Args:
    - lst (list): List of elements to generate combinations from.

    Returns:
    - list: List of tuples representing all possible combinations with at least two elements.
    """
    all_combinations = []
    for r in range(3, len(lst) + 1):
        all_combinations.extend(combinations(lst, r))
    return all_combinations

def add_evaluation_to_workbook(evaluation_df, workbook_path, macro_name="format_metrics", apply_macro=True):
    """
    Adds a pandas DataFrame containing evaluation metrics to an existing Excel workbook.

    Parameters:
    - evaluation_df (pd.DataFrame): DataFrame containing evaluation metrics.
    - workbook_path (str): File path to the existing Excel workbook.
    - macro_name (str, optional): Name of the macro in the workbook for formatting metrics. Default is "format_metrics".
    - apply_macro (bool, optional): Flag to apply the macro after adding the DataFrame. Default is True.

    Returns:
    - None

    Notes:
    - This function uses xlwings to interact with Excel. Make sure xlwings and pandas are installed.

    Example:
    add_evaluation_to_workbook(evaluation_df, "path/to/workbook.xlsx", macro_name="format_metrics", apply_macro=True)
    """
    try:
        # Load the existing workbook
        wb = xw.Book(workbook_path)

        # Add DataFrame to a new sheet
        ws = wb.sheets.add()
        ws.range("A1").value = evaluation_df

        # Run the macro to format the metrics if apply_macro is True
        if apply_macro:
            app = xw.apps.active
            app.api.Run(macro_name)

        # Save the workbook
        wb.save()

        # Make Excel application visible
        wb.app.visible = True

    except Exception as e:
        print(f"An error occurred while processing the workbook: {str(e)}")
    
def base_dataset():
    """
    Reads the base dataset after applying various filters based on predefined criteria.

    Returns:
    - pd.DataFrame: Filtered DataFrame containing the base dataset.
      Returns None if there are errors during filtering or if 'Thesis Default' column has NaN values.
    """
    try:
        df = df = pd.read_csv(FILE_PATH)

        # Filter LTV (Initial)
        df = df[(df['LTV (Initial)'] > 0) & (df['LTV (Initial)'] < 1)].copy()

        # Filter IC Combined (Initial)
        df = df[(df['IC Combined (Initial)'] > 0) & (df['IC Combined (Initial)'] < 10)].copy()

        # Filter EBITDA (Initial)
        df = df[df['EBITDA (Initial)'] >= 1e6].copy()

        # Filter EV Multiple (Initial)
        df = df[(df['EV Multiple (Initial)'] > 1) & (df['EV Multiple (Initial)'] < 30)].copy()

        # Filter Ownership and Security
        df = df[((df['Security'] == 'First Lien or Unitranche') |
                 (df['Security'] == 'Second Lien or Mezzanine')) &
                ((df['Ownership'] == 'Sponsor') |
                 (df['Ownership'] == 'Private Corporate'))].copy()

        # Filter Region
        df = df[(df['Region'] == 'US') | (df['Region'] == 'Europe')].copy()

        # Filter Total Net Leverage (Initial)
        df = df[(df['Total Net Leverage (Initial)'] > 2) & (df['Total Net Leverage (Initial)'] < 10)].copy()

        # Drop rows with NaN in 'Thesis Default'
        df = df.dropna(subset=['Thesis Default'])

        return df
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def domain_filter(df0):
    """
    Filters the given DataFrame `df0` based on predefined criteria.

    Args:
    - df0 (pd.DataFrame): Original DataFrame to filter.

    Returns:
    - pd.DataFrame: Filtered DataFrame containing the club 1 base dataset.
      Returns None if 'Thesis Default' column has NaN values after filtering.
    """
    try:
        df = df0.copy()

        # Filter LTV (Initial)
        df = df[(df['LTV (Initial)'] > 0) & (df['LTV (Initial)'] < 1)].copy()

        # Filter IC Combined (Initial)
        df = df[(df['IC Combined (Initial)'] > 0) & (df['IC Combined (Initial)'] < 10)].copy()

        # Filter EBITDA (Initial)
        df = df[df['EBITDA (Initial)'] >= 1e6].copy()

        # Filter EV Multiple (Initial)
        df = df[(df['EV Multiple (Initial)'] > 1) & (df['EV Multiple (Initial)'] < 30)].copy()

        # Filter Total Net Leverage (Initial)
        df = df[(df['Total Net Leverage (Initial)'] > 2) & (df['Total Net Leverage (Initial)'] < 10)].copy()

        # Filter Security
        df = df[(df['Security'] == 'First Lien or Unitranche') | 
                (df['Security'] == 'Second Lien or Mezzanine')].copy()

        return df
    

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def process_loan_portfolio(features, DATA_NAME, model, trained_scaler, CLUSTERED, confidence_interval=False,):
    """
    Process a portfolio of loans and compute the probability of default for each loan plus uncertainty interval for bayesian logistic regression model; ID column needs to be provided for efficient mapping 

    Parameters:
    features (list): List of feature names.
    DATA_NAME (str): Name of the data file.
    model (object): Model to predict the probability of default.
    trained_scaler (object): Trained scaler for preprocessing.
    CLUSTERED (bool): Boolean flag for clustering.
    confidence_interval (bool): Boolean flag for contrcuting interval when using Bayesian inference 

    Returns:
    None
    """

    # Load data from the specified Excel file
    data = pd.read_excel(DATA_NAME)
    
    # Strip extra spaces from the column names
    data.columns = data.columns.str.strip()
    
    # Features to check in the dataset (excluding 'Thesis Default')
    features_to_check = [feat for feat in features if feat != 'Thesis Default']
    features_in_data = set(data.columns)

    # Ensure all required features are present in the dataset
    assert set(features_to_check).issubset(features_in_data), "Not all features are present in the dataset"

    # Apply pre-clustering function to 'Security' column
    data['Security'] = data['Security'].apply(Security_PreCluster)
    
    # Filter the data based on domain criteria
    filtered_data = domain_filter(data)
    
    # Preprocess the filtered data
    processed_data, _ = pre_processing(
        filtered_data, 
        features=features_to_check, 
        trained_scaler=trained_scaler, 
        simply_clustered=CLUSTERED,
    )
    
    # Ensure the shapes of filtered and processed data match
    assert filtered_data.shape[0] == processed_data.shape[0], "Shape mismatch between filtered and processed data"
    
    # Compute the probability of default for each loan
    filtered_data['Probability of Default'] = model.predict_proba(processed_data)  

    # Initialize 'Probability of Default' with NaN
    data['Probability of Default'] = np.nan

    assert 'ID' in data.columns, 'No IDs provided'

    # drop duplicates
    filtered_data.drop_duplicates(subset='ID', keep='first', inplace=True,)

    # Create a mapping from 'ID' to 'Probability of Default'
    prob_default_map = filtered_data.set_index('ID')['Probability of Default']
    
    # Map the 'Probability of Default' back to the original data
    data['Probability of Default'] = data['ID'].map(prob_default_map)

    if confidence_interval:
        assert model.classifier_method == 'bayesian_logistic_regression', "Confidence interval for default probabilities only possible when using Bayesian inference."

        w_map, cov, bias = model.blr_param['w_map'].reshape(-1, 1), model.blr_param['cov_matrix'], model.blr_param['bias']

        print(f"Maximum a posterior estimate: {w_map}")
        print(f"Standard deviations of weight components w.r.t. posterior estimate: {np.diag(cov)}")
        df_test = processed_data.copy()
        for feature in [x for x in SECOND_ORDER if x in features]:           
            higher_order_feature = f"{feature}**2"
            df_test[higher_order_feature] = df_test[feature]**2
    
        # Add mixing terms
        if MIXING_TERMS:
            for feature_i, feature_j in combinations([x for x in SECOND_ORDER if x in features], 2):
                mixed_feature = f'{feature_i} * {feature_j}'
                df_test[mixed_feature] = df_test[feature_i] * df_test[feature_j]
            
        for feature in [x for x in THIRD_ORDER if x in features]:
            higher_order_feature = f"{feature}**3"
            df_test[higher_order_feature] = df_test[feature]**3
        
        df_test.sort_index(axis=1)
        X = df_test.values
        std = (np.diag(X @ cov @ X.T)).reshape(-1, 1)
        mean  = (X @ w_map).reshape(-1, 1)
        l_prob_of_default = sigmoid(mean - 1/2 *  std + bias)
        r_prob_of_default = sigmoid(mean + 1/2 *  std + bias)

        # Compute the probability of default for each loan
        filtered_data['Lower Probability of Default'] = l_prob_of_default
        filtered_data['Upper Probability of Default'] = r_prob_of_default
        # Initialize 'Probability of Default' with NaN
        data['Lower Probability of Default'] = np.nan
        data['Upper Probability of Default'] = np.nan

        # Create a mapping from 'ID' to 'Probability of Default'
        prob_default_map = filtered_data.set_index('ID')['Lower Probability of Default']
        data['Lower Probability of Default'] = data['ID'].map(prob_default_map)
        prob_default_map = filtered_data.set_index('ID')['Upper Probability of Default']
        data['Upper Probability of Default'] = data['ID'].map(prob_default_map)

    # Save the final data with probability of default to an Excel file
    data.to_excel(DATA_NAME[:-5] + '_with Probability_of_Default.xlsx')


if __name__ == "__main__":

    # Extract filtered dataset
    df = base_dataset()
    print(f"Dataset size: {df.shape[0]}")
    print(f"Default Rate: {np.round(100*df['Thesis Default'].mean(), 2)}%")

    if GENERATE_FILTERED_DATASET:
        df.to_excel("Thesis Filtered Dataset.xlsx", index=False)
        print("Stored the new Dataset of for the Thesis project!")

    df_base = df.copy()
    default_rate = df_base['Thesis Default'].mean()

    # Get the feature combos one wants to test; either on tests all the features or all subcombinations according to 
    # generate_combinations
    if SPECIFIC_FEATURE_COMBO[0]:
        # only one combo
        all_feature_combos = [SPECIFIC_FEATURE_COMBO[1] + ['Thesis Default']]
    else:
        # all combos
        all_feature_combos = [list(combos) + ['Thesis Default'] for combos in generate_combinations(FEATURES)]

    all_feature_combos.reverse()

    # Metrics one wants to use; one only evauluates if num of training iterations is larger than 1
    if COMPUTE_COFUSION_METRICS and (NUM_OF_TRAINING > 1):

        indices = ['AUC', f'pAUC', 'BCE', 'Brier',]
        # Add evaluation metrics indices for each threshold
        for threshold in THRESHOLDS:
            indices += [
                f'Precision',
                f'Recall',
                f'FPR',
                f'F1/2',
                f'F1',
                f'F2',
            ]
        indices += ["PAM", "Chosen Threshold",]

        # Initialize output dictionary
        out_final = {}

    # Loop through all the possible chosen feature combinations
    for i, features in enumerate(all_feature_combos):
        
        print(f"Feature Combo [{i+1}/{len(all_feature_combos)}]")

        features.sort()

        # collect higher order terms one wants to use
        second_order = [feat for feat in SECOND_ORDER if feat in features]
        third_order = [feat for feat in THIRD_ORDER if feat in features]

        
        df = df_base.copy()
        # apply preprocessing and collect the scaler which needs to to be applied later for any prediction
        df, trained_scaler = pre_processing(df, features=features, simply_clustered=CLUSTERED, scaling=SCALING, print_weights=PRINT_WEIGHTS,)
        

        assert df.shape[0] == df_base.shape[0], "Dataset has lost rows! Further Filtering has been applied to Master Dataset"

        # loop thorugh all the models one wants to test; generally advised to use one model 
        for clustering_method, classifier in CLASSIFIERS:
            
            # store the feature model name (feature combo + classifier used)
            name = '/'.join([elem for elem in features if elem != 'Thesis Default'])
            model_str = f'{name} {classifier}'
            
            # no clustering
            if clustering_method != 'continuous':
                model_str += (' ' + clustering_method) 

            # clustering with specified floors and caps in pre_processing 
            if CLUSTERED:
                model_str += " (clustered) "

            # mixxing terms for second order terms
            if MIXING_TERMS:
                model_str += " (with mixed terms) "
            
            # if hypertuning is applied 
            if HYPERTUNE:
                model_str += "HT"
            
            # if smote is applied to underlying dataset
            if SMOTE:
                model_str = "(smote)" + model_str 

            if (NUM_OF_TRAINING > 1) and COMPUTE_COFUSION_METRICS:
                # Collect output vector to compute the averages  
                out = []

                # Collect average FPR and TPR for designed threshold for project specific measure (PAM)
                avr_fpr = []
                avr_tpr = []
 
            if classifier in ['mean']:
                    df = df_base.copy()

            # Collect for each traing round the optimal hyperparameter
            if HYPERTUNE and (classifier in  ['logistic_regression_higher_order', 'SGD_classifier_higher_order']):
                optimal_hyperparamters = {
                                        'l2':  {
                                                'C': []
                                                },
                                        'l1':  {
                                                'C': [],
                                                },
                                        'elasticnet': {
                                                'C': [],
                                                'l1_ratio': [],
                                                },
                                    }[REGULARIZER]        

            # loop through number of training procedures
            for j in range(NUM_OF_TRAINING):

                if classifier == 'nn_classifier' or HYPERTUNE:
                    print(f"Training Procedure: {j+1}/{NUM_OF_TRAINING}" )

                # Radom Splits
                if NUM_OF_TRAINING > 1:
                    df_train, df_test, y_train, y_test = train_test_split(df.drop(columns=['Thesis Default']), df['Thesis Default'], test_size=0.2, random_state=125+j) 
                    df_train['Thesis Default'] = y_train
                    df_test['Thesis Default'] = y_test
                
                # Historical Split or whole dataset used for training
                else:
                    df = base_dataset()
                    df = df.sort_values(by='Launch Date')
                    if not (classifier in ['mean']): 
                        df, _ = pre_processing(df, features=features, simply_clustered=CLUSTERED, trained_scaler=trained_scaler, scaling=SCALING, )

                    split_index = int(TRAIN_SIZE * len(df))
                    df_train = df.iloc[:split_index]

                model = Feature_Model(df_train, 
                                    classifier=classifier, 
                                    clustering_method=clustering_method,
                                    features=features,
                                    simply_clustered=CLUSTERED, second_order_features=second_order, third_order_features=third_order, penalty=REGULARIZER,
                                    alpha=REGULARIZATION_STRENGTH, num_of_epochs=NUM_OF_EPOCHS, mixing_terms=MIXING_TERMS, optimizer=OPTIMIZER, l1_ratio=l1_ratio,
                                    hypertune=HYPERTUNE, scoring=SCORING, smote=SMOTE, random_state=j+777, print_weights=PRINT_WEIGHTS, 
                                    )
                model.fit()


                # collect optimal hyperparameters if hypertuningis being applied; only for higher order logistic regression model available
                if HYPERTUNE and  (classifier in  ['logistic_regression_higher_order', 'SGD_classifier_higher_order']):
                    best_parameters = model.optimal_hyper_parameters 
                    for key in optimal_hyperparamters.keys():
                        optimal_hyperparamters[key].append(best_parameters[key])
                        
                # TODO: modle not used remove potentially 
                if model.classifier_method == 'regularized_logistic_regression':
                    print(f'Best Regularization Parameter: {np.round(1/model.classifier.C_, 0)}')

                if NUM_OF_TRAINING == 1:

                    # Collecting the weight distribution seperately by making use of bootstrapping; checks significance basically 
                    if GET_WEIGHTS and (classifier in ['logistic_regression', 'logistic_regression_higher_order',]):

                        # Create a dataframe to store feature names and their corresponding weights
                        weights_df = pd.DataFrame({'Feature': model.df.drop('Thesis Default', axis=1).columns, 'Weight': model.classifier.coef_[0]})

                        # Define the number of bootstrap samples
                        num_bootstraps = 1000
                        weights = (weights_df['Weight']).values
                        # Initialize an array to store the bootstrap coefficients
                        bootstrap_coefs = np.zeros((num_bootstraps, len(weights)))

                        X = model.X
                        y = model.y
                        test_model = model.classifier

                        # Perform bootstrapping
                        for i in range(num_bootstraps):
                            # Resample with replacement from the original data
                            bootstrap_indices = np.random.choice(range(len(X)), len(X), replace=True)
                            X_bootstrap = X[bootstrap_indices]
                            y_bootstrap = y[bootstrap_indices]
                            
                            # Fit logistic regression model on the bootstrap sample
                            test_model.fit(X_bootstrap, y_bootstrap)
                            
                            # Store the coefficients
                            bootstrap_coefs[i] = test_model.coef_

                        mean_weights = []
                        std_weights = []
                        median_weights = []
                        lower25 = []
                        upper25 = []
                        for i in range(len(weights)):
                            coef = weights[i]
                            coef_bootstraps = bootstrap_coefs[:, i]
                            
                            # Compute the proportion of bootstrap samples where the coefficient is greater than 0
                            mean_weights.append(coef_bootstraps.mean())
                            std_weights.append(coef_bootstraps.std())
                            median_weights.append(np.percentile(coef_bootstraps, 50))
                            lower25.append(np.percentile(coef_bootstraps, 25))
                            upper25.append(np.percentile(coef_bootstraps, 75))

                        weights_df['Weight Mean'] = mean_weights
                        weights_df['Weight 25 Perc.'] = lower25    
                        weights_df['Weight Median'] = median_weights
                        weights_df['Weight 75 Perc.'] = upper25
                        weights_df['Weight Std'] = std_weights

                        add_evaluation_to_workbook(weights_df, workbook_path=EVAL_PATH, apply_macro=False)

                    if SPECIFIC_DATASET:
                        process_loan_portfolio(features=features, DATA_NAME=DATA_NAME, model=model, trained_scaler=trained_scaler, CLUSTERED=CLUSTERED, confidence_interval=UNCERTAINTY_INTERVAL)
                        print("Specific dataset with default probability has been generated!")

                    # generate all heatmap combos
                    if FEATURE_PLOTS:
                        for two_features in combinations([x for x in features if x != 'Thesis Default'], 2):
                            model.plot_heatmap(features=list(two_features), trained_scaler=trained_scaler, scaling=SCALING, )
                    
                    if SINGLE_FEATURE_PLOTS:
                        single_feature_list = [[feat] for feat in features if feat != 'Thesis Default']
                        for feat in single_feature_list:
                            model.plot_feature_pd(feature=feat, trained_scaler=trained_scaler, scaling=SCALING, convert_to_lossrate=CONVERT_TO_LOSSRATE, recovery_rate=RECOVERY_RATE, sub_delta=SUB_DELTA, )

                    # only show evaluation plots if there is a test set
                    if EVALUATION_PLOTS and (TRAIN_SIZE < 1):
                        df_test  = df_train[split_index:]
                        y_test = df_test['Thesis Default']
                        prob_of_default = model.predict_proba(df_test)
                        model.plot_auc(y_true=y_test, y_score=prob_of_default)
                        model.plot_calibration_curve(y_true=y_test, y_prob=prob_of_default,n_bins=N_BINS, y_lim=Y_LIM, x_lim=X_LIM,)
                        model.plot_precision_recall_curve(y_true=y_test, y_score=prob_of_default)

                # Collect single metrics   
                if (NUM_OF_TRAINING>1) and COMPUTE_COFUSION_METRICS:
                    prob_of_default = model.predict_proba(df_test)
                   
                    # add all standard evaluation metrics
                    out.append(model.evaluation_metrics(y_true=y_test, y_score=prob_of_default, thresholds=THRESHOLDS, fpr_range=AUC_RANGE, ))
                    new_fprs_per_threshold = []
                    new_tprs_per_threshold = []
                    for threshold in BASE_CASE_THRESHOLDS:
                        y_pred = (prob_of_default >= threshold).astype(int)
                        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                        fpr = fp / (fp + tn) if (tn + fp) > 0 else float('nan')
                        tpr = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
                        new_fprs_per_threshold.append(fpr)
                        new_tprs_per_threshold.append(tpr)
                    avr_fpr.append(new_fprs_per_threshold)
                    avr_tpr.append(new_tprs_per_threshold)

            # Averaged metrics for evaluation file
            if COMPUTE_COFUSION_METRICS and (NUM_OF_TRAINING > 1):
                avr_fpr = np.array(avr_fpr).mean(axis=0)
                avr_tpr = np.array(avr_tpr).mean(axis=0)
                cum_net_r = base_case_cum_net_return(fprs=avr_fpr, tprs=avr_tpr, thresholds=BASE_CASE_THRESHOLDS, loan_life=BASE_CASE_LOAN_LIFE, default_rate=default_rate, ) 
                # Add metrics and project specific measure
                out_final[model_str] = list(np.mean(np.array(out), axis=0))+ [cum_net_r[0], cum_net_r[1]]

            if HYPERTUNE and  (classifier in  ['logistic_regression_higher_order', 'SGD_classifier_higher_order']): 

                for key in optimal_hyperparamters.keys():
                    param_values = optimal_hyperparamters[key]
                    value_counts = Counter(param_values)  # Count occurrences of each value
                    
                    # Sort the counts in descending order
                    sorted_value_counts = sorted(value_counts.items(), key=lambda x: x[1], reverse=True)
                    
                    print(f"Parameter {key}:")
                    for value, count in sorted_value_counts:
                        print(f"    Value: {value}, Count: {count}")
          
    if COMPUTE_COFUSION_METRICS and (NUM_OF_TRAINING > 1):            
        evaluation_df = pd.DataFrame(out_final, index=indices).T
        print("Creating excel file with evaluation metrics..")
        evaluation_df.to_excel("Model Evaluation.xlsx")
        # print(f"Test set default count and average default rate: {df_test['Thesis Default'].shape[0]}, {np.round(df_test['Thesis Default'].mean() * 100, 2)}%")
        add_evaluation_to_workbook(evaluation_df=evaluation_df, workbook_path=EVAL_PATH, macro_name="format_metrics")

    