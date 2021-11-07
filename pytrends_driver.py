import pip


def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])


install('mlflow')
install('pandas')
install('pytrends')

import mlflow
import mlflow.keras
import mlflow.tracking
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Sequential

from pytrends.request import TrendReq
from util import lines_to_word_map, parse_args, read_file, write_csv, write_json


def main():
    # get arguments from user
    args = parse_args()

    # process input keywords
    lines = read_file(args.input_path)
    word_map = lines_to_word_map(lines)
    print(word_map)

    # save word map to JSON
    output_stem = args.output_path.split('.')
    output_json = output_stem[0] + ".json"
    write_json(word_map, output_json)

    # get Google Trend info for each keyword
    iot_dict = trends_for_word_map(word_map)
    predictions = pd.DataFrame()

    # flatten trends for different categories into single list of tuples
    interest_tuples = interest_at_timestamp(iot_dict)

    # convert list of tuples into Pandas dataframe, and write to CSV
    interest_df = interest_tuples_to_df(interest_tuples)

    # generate prediction for each model and write to dataframe
    for keyword in interest_df.columns():
        training_set, test_set, scalar = train_test_split(interest_df, keyword)
        model = model_train(training_set, keyword)
        prediction = model_predict(test_set, model, keyword)
        predictions = pd.concat([predictions, prediction], axis=1)

    # write results to output path
    predictions.to_csv(args.output_path)

    return 0


def trends_for_word_map(word_map):
    iot_dict = {}
    for category, keywords in word_map.items():
        print("Category: " + category)
        print("Keyword(s): " + keywords)
        iot_df = get_trends(keywords)
        iot_dict[category] = iot_df
    return iot_dict


def interest_at_timestamp(iot_dict):
    interest_tuples = []
    for category, iot_df in iot_dict.items():
        iot_df.drop(columns=['isPartial'], axis=1, inplace=True)
        for rowIndex, row in iot_df.iterrows():
            for colIndex, value in row.items():
                interest_tuple = (rowIndex, colIndex, category, value)
                interest_tuples.append(interest_tuple)
    return interest_tuples


def interest_tuples_to_df(interest_tuples):
    interest_df = pd.DataFrame(interest_tuples, columns=['Date', 'Keyword', 'Category', 'Interest'])
    interest_df.sort_values(by=['Keyword', 'Date', 'Category'], inplace=True)
    return interest_df


def get_trends(keywords):
    trend = TrendReq()
    trend.build_payload(kw_list=keywords)
    iot = trend.interest_over_time()
    iot_df = pd.DataFrame(iot)
    return iot_df


def train_test_split(df, keyword, prediction_weeks=25):
    split = len(df) - prediction_weeks
    df_train = df[:split][keyword]
    df_test = df[split:][keyword]

    training_set, test_set = df_train.values, df_test.values
    training_set = np.reshape(training_set, (len(training_set), 1))

    sc = MinMaxScaler()
    training_set = sc.fit_transform(training_set)
    X_train = training_set[0: len(training_set) - 1]
    y_train = training_set[1: len(training_set)]
    X_train = np.reshape(X_train, (len(X_train), 1, 1))
    training_set = (X_train, y_train)

    test_set = np.reshape(test_set, (len(test_set), 1))
    test_set = sc.transform(test_set)
    test_set = np.reshape(test_set, (len(test_set), 1, 1))

    return training_set, test_set, sc


def model_train(training_set, keyword, optimizer='adam', loss='mean_squared_error',
                batch_size=5, epochs=150):
    X_train, y_train = training_set
    # define the model
    model = Sequential()
    # input and dense layers
    model.add(LSTM(units=4, activation='sigmoid', input_shape=(None, 1)))
    # output layer
    model.add(Dense(units=1))

    # compile and train model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    # log model output with MLFlow
    with mlflow.start_run() as run:
        mlflow.keras.log_model(model, 'Google Trend Prediction: ' + keyword)

    return model


def model_predict(test_set, model, scalar):
    predictions = model.predict(test_set)
    predictions = scalar.inverse_transform(predictions)
    return predictions


if __name__ == '__main__':
    main()
