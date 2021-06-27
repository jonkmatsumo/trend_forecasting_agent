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


def main():
    # get arguments from user
    args = parse_args()

    # process input keywords
    keywords = read_file(args.input_path)
    print("Input Keywords:")
    print(keywords)

    # get Google Trend info for each keyword, write to CSV
    trend_df = get_trends(keywords)
    trend_df.to_csv(args.output_path)
    print("\nFinal Result: ")
    print(trend_df.head())

    return 0


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_path", help="path to read in keywords from")
    arg_parser.add_argument("output_path", help="path to write trend results to")
    return arg_parser.parse_args()


def read_file(filename):
    file = open(filename, 'r')
    lines = file.read().splitlines()
    file.close()
    return lines


def get_trends(keywords):
    pytrend = TrendReq()
    pytrend.build_payload(kw_list=keywords)
    iot = pytrend.interest_over_time()
    iot_df = pd.DataFrame(iot)
    return iot_df


if __name__ == '__main__':
    main()
