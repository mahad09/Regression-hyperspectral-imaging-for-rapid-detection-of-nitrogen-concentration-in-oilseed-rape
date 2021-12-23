import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.feature_selection import f_regression, SelectKBest
from sklearn.metrics import mean_absolute_error, explained_variance_score, mean_squared_error, r2_score, roc_auc_score, plot_roc_curve, RocCurveDisplay
from tqdm import tqdm
import pickle
from yellowbrick.regressor import PredictionError
from pandas_profiling import ProfileReport
import pdb
import matplotlib
matplotlib.use( 'tkagg' )

RANDOM_FOREST_REGRESSOR_FILENAME = 'random_forest_regressor.sav'
DECISION_TREE_REGRESSOR_FILENAME = 'decision_tree_regressor.sav'
SUPPORT_VECTOR_MACHINE_FILENAME = 'svm_regressor.sav'
# tomer.porat@outlook.com

# loading the whole dataset from the given csv file.
def load_and_analyze_dataset(file_path):
  df = pd.read_csv(file_path)

  # Giving an insight about the dataset using the following lines.
  print('Dataset shape: ', df.shape)
  print('Dataset information: ', df.info())
  print('Dataset spreadness: ')
  print(df.describe())

  feature_names = df.columns.values.tolist()
  feature_names.remove('N Values') 

  # profile = ProfileReport(df, minimal=True)

  # Uncomment the following lines for EDA. It will take some time for execution of below 2 lines.
  # profile.to_file(output_file="dataset_report.html")
  # profile.to_file(output_file="dataset_report.json")

  return df, feature_names


# finding and removing duplicates from each column.
def remove_duplicates(df):
  duplicates = df.duplicated()

  if duplicates.sum() != 0:  
    df.drop_duplicates(inplace=True)

  return df


# Removing the rows containing any missing value.
def remove_missing_values(df):
  df = df.dropna(axis=0, how='any')

  return df


# finding and removing outliers from each column.
def remove_outliers(df):
  feature_columns = df.columns.values.tolist()[3:]

  for feature_column in feature_columns:
    lower_bound = df[feature_column].quantile(0.01)
    upper_bound = df[feature_column].quantile(0.99)

    [df[(df[feature_column] < upper_bound) & (df[feature_column] > lower_bound)]]

  return df


# The dataset is splitted based on the 'Dataset' column. 'Dataset=1' goes to training set and
# the 'Dataset=0' goes into test set. 
def dataset_splitting(df):
  x_train = df.loc[df['Dataset']==1].iloc[:, df.columns!='N Values'].values
  x_test = df.loc[df['Dataset']==0].iloc[:, df.columns!='N Values'].values

  y_train = df.loc[df['Dataset']==1]['N Values'].values
  y_test = df.loc[df['Dataset']==0]['N Values'].values

  return x_train, y_train, x_test, y_test


# Applying feature selection on the dataset.
def feature_selection(x_train, y_train, x_test, feature_names):
  # selecting the best k number of features by change k parameter.
  fs = SelectKBest(score_func=f_regression, k=350)
  fs.fit(x_train, y_train)

  x_train_fs = fs.transform(x_train)
  x_test_fs = fs.transform(x_test)

  for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))

  selected_k_features = []

  mask = fs.get_support()

  for is_selected, feature in zip(mask, feature_names):
    if is_selected:
        selected_k_features.append(feature)

  print('selected features: ')
  print(selected_k_features)

  return x_train_fs, x_test_fs


def decision_tree_regressor(x_train, y_train):
  regressor = DecisionTreeRegressor(max_depth=10, random_state=0)

  print('Decision tree regressor Information:')
  print(regressor.get_params())

  regressor.fit(x_train, y_train)

  pickle.dump(regressor, open(DECISION_TREE_REGRESSOR_FILENAME, 'wb'))

  return regressor


def random_forest_regressor(x_train, y_train):
  regressor = RandomForestRegressor(max_depth=20, verbose=1, n_jobs=-1)

  print('Random forest regressor Information:')
  print(regressor.get_params())

  regressor.fit(x_train, y_train)

  pickle.dump(regressor, open(RANDOM_FOREST_REGRESSOR_FILENAME, 'wb'))

  return regressor


def support_vector_machine_regressor(x_train, y_train):
  regressor = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2, max_iter=50000, verbose=True))

  print('SVM Information:')
  print(regressor.get_params())

  regressor.fit(x_train, y_train)

  pickle.dump(regressor, open(SUPPORT_VECTOR_MACHINE_FILENAME, 'wb'))

  return regressor


def model_evaluation(x_test, y_test, regressor):
  y_predict = regressor.predict(x_test)

  accuracy = regressor.score(x_test, y_test)
  print('Accuracy: ', accuracy*100)

  variance_score = explained_variance_score(y_test, y_predict)
  print('Explained variance score: ', variance_score)

  r2_accuracy = r2_score(y_test, y_predict)
  print('r2 score: ', r2_accuracy)

  mae = mean_absolute_error(y_test, y_predict)
  print('Mean absolute error: %.3f' % mae)

  mse = mean_squared_error(y_test, y_predict)
  print('Mean squared error: %.3f' % mse)



def plot_prediction_error(x_test, y_test, regressor):
  visualizer = PredictionError(regressor)
  visualizer.score(x_test, y_test)
  visualizer.show(outpath=visualizer.name)



df, feature_names = load_and_analyze_dataset('Meanspectra.csv')

df = remove_duplicates(df)
df = remove_missing_values(df)
df = remove_outliers(df)

x_train, y_train, x_test, y_test = dataset_splitting(df)
x_train, x_test = feature_selection(x_train, y_train, x_test, feature_names)

# Uncomment one of the following commented model lines to for training. 
# model = decision_tree_regressor(x_train, y_train)
model = random_forest_regressor(x_train, y_train)
# model = support_vector_machine_regressor(x_train, y_train)

model_evaluation(x_test, y_test, model)
plot_prediction_error(x_test, y_test, model)


