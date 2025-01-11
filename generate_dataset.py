import pandas as pd 
import numpy as np 


num_of_loans = 3000
default_rate = 3.81/100

df =  pd.DataFrame()

df['ID'] = list(range(num_of_loans))
count_per_range = [int(num_of_loans*0.1), int(num_of_loans*0.15), int(num_of_loans*0.25), int(num_of_loans*0.25), int(num_of_loans*0.15)]
count_per_range.append(num_of_loans - sum(count_per_range))
assert sum(count_per_range) == num_of_loans
print(sum(count_per_range))
higher_bounds = {
    'EBITDA (Initial)': {
        'upper_bounds': [x*1e6 for x in [8.2, 16.5, 35.4, 66.0, 123.5, 300.0]],
        'start': 1.0,
    },
    'EV Multiple (Initial)': {
        'upper_bounds': [5.45, 6.64, 8.00, 9.61, 11.75, 30.0],
        'start': 1.0,
    },
    'IC Combined (Initial)': {
        'upper_bounds': [1.77, 2.22, 2.79, 3.61, 4.47, 10],
        'start': 0.1,
    },
    'LTV (Initial)': {
        'upper_bounds': [x / 100 for x in [32.3, 39.8, 48.5, 57.8, 66.2, 99.0]],
        'start': 0.1,
    },
    'Total Net Leverage (Initial)': {
        'upper_bounds': [3.01, 3.75, 4.51, 5.42, 6.23, 10.0],
        'start': 2.0,
    },
}
# add numeric columns
for column in list(higher_bounds.keys()):
    values = []
    lower_bound = higher_bounds[column]['start']
    for i, (count, upper_bound) in enumerate(zip(count_per_range, higher_bounds[column]['upper_bounds'])):
        values.extend(list(np.linspace(start=lower_bound, stop=upper_bound, num=count)))
        lower_bound = upper_bound
    if column == 'EBITDA (Initial)' or column == 'EV Multiple (Initial)':
        values = list(reversed(values))
    df[column] = values

# add security type
values = ['First Lien or Unitranche' for i in range(int(num_of_loans*0.8))]
values.extend(['Second Lien or Mezzanine' for i in range(num_of_loans - int(num_of_loans*0.8))])
df['Security'] = values

# add default boolean
probabilities = np.linspace(0.01, default_rate * 2, num=num_of_loans)  # Increase probability over loans
df['Thesis Default'] = np.random.binomial(1, probabilities)

# add dataset 
df.to_csv('Your_Dataset.csv', index=False)
# add loan portofolio 
df.sample(n=20).to_excel('loan_portfolio.xlsx', index=False)

# new_df = pd.read_csv(filepath_or_buffer='Your_Dataset.csv')