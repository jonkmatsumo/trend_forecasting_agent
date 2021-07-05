import pip

def install(package):
    if hasattr(pip, 'main'):
        pip.main(['install', package])
    else:
        pip._internal.main(['install', package])


install('pandas')
install('pytrends')

import argparse
import pandas as pd
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

    # flatten trends for different categories into single list of tuples
    interest_tuples = interest_at_timestamp(iot_dict)

    # convert list of tuples into Pandas dataframe, and write to CSV
    interest_df = interest_tuples_to_df(interest_tuples)
    write_csv(interest_df, args.output_path)

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
    interest_df = pd.DataFrame(interest_tuples, columns=['Date','Keyword','Category','Interest'])
    interest_df.sort_values(by=['Keyword', 'Date', 'Category'], inplace=True)
    return interest_df


def get_trends(keywords):
    pytrend = TrendReq()
    pytrend.build_payload(kw_list=keywords)
    iot = pytrend.interest_over_time()
    iot_df = pd.DataFrame(iot)
    return iot_df


if __name__ == '__main__':
    main()
