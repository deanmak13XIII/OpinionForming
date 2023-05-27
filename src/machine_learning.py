import ast
import logging
import os
import re

import numpy
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

from utils.constants import DATA_DIR
from utils.graph_generator import read_graph_data_from_csv
from utils.utils import get_unique_or_latest_file_name

logging.basicConfig(level=logging.INFO)
_logger = logging.getLogger("OpinionForming")


def engineer_data(data: pd.DataFrame):
    """
    Formats the data as read from a csv, by expanding and removing unnecessary columns
    :param data: The graph data as read from csv. In csv format
    """
    # Find single value to represent the columns with multiple values
    # Drop all columns with multiple values [(1. Calc (av./min/max) closeness in GD.), (2. Calc (av./min/max) node betweenness in GD), (3. )]
    data = data.drop(
        columns=['matrix', 'colour_states', 'node_degrees', 'node_eccentricities', 'node_closeness',
                 'node_betweenness'])

    _logger.info("Expanding compressed columns...")
    # Separate each (initial_colour_decimal | colour_steps | colour_loop_back | colour_state_ratios) into a row of it's own. Duplicate all other values into that row
    index = 0
    new_all_data_dict = {}
    # Retrieving and applying column names to be keys in new all_data dict
    for col_name in data.columns.tolist():
        new_all_data_dict[col_name] = []
    # Going through each row and expanding colour related lists so each value takes a row of its own
    for initial_colour_list_str in data['initial_colour_decimal']:
        initial_colour_list = ast.literal_eval(initial_colour_list_str)
        for col_name in data.columns.tolist():
            cell = data[col_name][index]
            if type(cell) == str and cell[0] == '[':
                cell = ast.literal_eval(cell)
            # if col_name in ('colour_steps', 'colour_loop_back_steps', 'colour_state_change_ratios'):
            if type(cell) == list and len(cell) == len(initial_colour_list):
                new_all_data_dict[col_name] += cell
            else:
                new_all_data_dict[col_name] += [cell, ] * len(initial_colour_list)
        index += 1
    data = pd.DataFrame(new_all_data_dict)
    return data


def prepare_data():
    """
    Cleans up all the data from the latest csv. Creates training and test data as well.
    """
    file_name = f"expanded_rf_graph_data_1.csv"
    file_path = os.path.abspath(os.path.join(DATA_DIR(), file_name))
    latest_file_path = get_unique_or_latest_file_name(file_path=file_path, latest=True)
    all_data = pd.DataFrame()
    if not os.path.isfile(latest_file_path):
        _logger.info("Reading latest graph_data [unexpanded]...")
        all_data: pd.DataFrame = read_graph_data_from_csv(entire_dataframe=True)
        all_data = engineer_data(all_data)
        _logger.info(f"Writing expanded graph data to csv: {file_path}")
        all_data.to_csv(file_path, index=False)
    else:
        _logger.info(f"Reading latest expanded graph data csv: {latest_file_path}")
        for chunk in pd.read_csv(latest_file_path, chunksize=50000, index_col=False):
            all_data = pd.concat([all_data, chunk])

    _logger.info("Proceeding to split ALL data into TRAIN and TEST data...")
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for label, group in all_data.groupby('vertices'):
        train, test = train_test_split(group, train_size=0.8, random_state=42)
        train_data = pd.concat([train_data, train])
        test_data = pd.concat([test_data, test])

    return all_data, train_data, test_data


def fit_model(dependent_variable: str, train_data):
    """
    Fits the model if one fitting the required needs is not already saved.
    """
    # Getting the vertices of the model. If v in data is higher, training new rf model. Else reusing saved model
    rf_fit_vertices = [int(re.search('\d+', file).group(0)) for file in os.listdir(DATA_DIR()) if file.startswith(f"rffit_{dependent_variable}") and re.search('\d+', file)]
    _logger.info(f"Train Data [max_vcount = {max(train_data['vertices'])}]...")
    model_vcount_highest = [True if max(train_data['vertices']) < vcount else False for vcount in rf_fit_vertices]
    file_name = f"rffit_{dependent_variable}_v{max(rf_fit_vertices)}.joblib"
    file_path = os.path.abspath(os.path.join(DATA_DIR(), file_name))

    # If it's all true that any saved model has the higher vcount. Not training unless train data available.
    if False not in model_vcount_highest or not model_vcount_highest and os.path.isfile(file_path):
        _logger.info(f"Loading saved Random Forest Model: {file_name}")
        rf = load(file_path)
        return rf
    y_train = train_data[dependent_variable]
    x_train = train_data.drop(dependent_variable, axis=1)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, verbose=1, n_jobs=-1)
    _logger.info("Fitting Random Forest Model...")
    rf.fit(x_train, y_train)
    _logger.info("Saving Random Forest Model to File...")
    dump(rf, filename=file_path)
    return rf


def predict(random_forest_model, dependent_variable, test_data):
    x_test = test_data.drop(dependent_variable, axis=1)
    _logger.info(f"Predicting {dependent_variable}...")
    y_pred = random_forest_model.predict(x_test)
    if dependent_variable in test_data:
        y_test = test_data[dependent_variable]
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = numpy.sqrt(mse)
        error_dict = {"mae": mae, "mse": mse, "rmse": rmse}
        return y_pred, error_dict
    return y_pred, None


def start_ml(dependent_variable: str = 'colour_steps', wanted_test_data: pd.DataFrame = None, save_predicted_csv: bool = False):
    """
    If no test data is provided, simply taking from latest all_graph_data file
    """
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    all_data, train_data, test_data = prepare_data()
    if wanted_test_data:
        test_data = engineer_data(wanted_test_data)
    rf = fit_model(dependent_variable=dependent_variable, train_data=train_data)
    predicted_data, error_metrics = predict(random_forest_model=rf, dependent_variable=dependent_variable, test_data=test_data)

    # Rounding all predicted values if the actual data doesn't have any floats in it, and only ints.
    if not test_data[dependent_variable].apply(lambda x: isinstance(x, float)).any():
        predicted_data = [round(float(x)) for x in predicted_data]
    if save_predicted_csv:
        file_name = f"predicted_gdata_v{max(test_data['vertices'])}.csv"
        file_path = os.path.abspath(os.path.join(DATA_DIR(), file_name))
        comparison_df = test_data
        comparison_df[f"predicted_{dependent_variable}"] = predicted_data
        comparison_df.to_csv(file_path, index=False)

    _logger.info(f"THESE ARE THE PREDICTION ERROR METRICS: {error_metrics}")
    return predicted_data, error_metrics


if __name__ == "__main__":
    start_ml(save_predicted_csv=True)
