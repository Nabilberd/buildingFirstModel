import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
# Import the train_test_split function
from sklearn.model_selection import train_test_split
# imputation
from sklearn.impute import SimpleImputer

# get absolute error using max leaf nodes
def get_mae( model, train_X, val_X, train_y, val_y):
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# Path of the file to read
iowa_file_train = '../data/train.csv'
iowa_file_test = '../data/test.csv'

home_data_train = pd.read_csv(iowa_file_train)
home_data_test = pd.read_csv(iowa_file_test)

y = home_data_train.SalePrice
feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data_train[feature_columns]
X_test = home_data_test[feature_columns]

# split our data to trainning and validation models
train_X, val_X, train_y, val_y = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state = 1)

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
model_6 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=1)
model_7 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=1)
model_8 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=1)

models = [model_1, model_2, model_3, model_4, model_5, model_6, model_7, model_8]

# select the best model
for i in range(0, len(models)):
    error = get_mae(models[i], train_X, val_X, train_y, val_y)
    print("Model %d MAE: %d" % (i+1, error))

# Write loop to find the ideal tree size from candidate_max_leaf_nodes
scores = {model : get_mae(model, train_X, val_X, train_y, val_y) for model in models}
print("best_tree_size", round(scores.get(min(scores, key=scores.get))))

# Make copy to avoid changing original data (when imputing)
X_train_plus = train_X.copy()
X_valid_plus = val_X.copy()

# Get names of columns with missing values
cols_with_missing = [col for col in train_X.columns
                     if train_X[col].isnull().any()]

# Make new columns indicating what will be imputed
for col in cols_with_missing:
    X_train_plus[col + '_was_missing'] = X_train_plus[col].isnull()
    X_valid_plus[col + '_was_missing'] = X_valid_plus[col].isnull()

print(cols_with_missing)

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
imputed_X_valid_plus = pd.DataFrame(my_imputer.transform(X_valid_plus))

# Imputation removed column names; put them back
imputed_X_train_plus.columns = X_train_plus.columns
imputed_X_valid_plus.columns = X_valid_plus.columns

my_model = model_2

# Generate test predictions
preds_test = my_model.predict(X_test)

print("MAE from Approach 3 (An Extension to Imputation):")
print(get_mae(my_model,imputed_X_train_plus, imputed_X_valid_plus, train_y, val_y))

# Save predictions in format used for competition scoring
samplesubmission = pd.read_csv('../data/sample_submission.csv')
output = pd.DataFrame({'Id': samplesubmission.Id,
                       'SalePrice': preds_test})
output.to_csv('../data/submission.csv', index=False)