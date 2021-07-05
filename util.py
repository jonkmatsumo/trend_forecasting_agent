import pandas as pd
import json
import argparse


def lines_to_word_map(lines):
    d = {}
    cur_cat = ""  # initially none
    for line in lines:
        if not line:
            continue
        words = line.split(':')
        if words[0] == 'Category':
            cur_cat = words[1]
            d[cur_cat] = []
        elif words[0] == 'Keyword':
            # TO-DO: stop user if more than 5 keywords in list
            # OR split into multiple categories (requires rejoining later)
            d[cur_cat].append(words[1])
        else:
            print("Could not process: " + line)
    return d


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


def write_csv(dataframe, output_path, index=False):
    dataframe.to_csv(output_path, index=index)


def write_json(dict, output_path):
    json_object = json.dumps(dict, indent=4)
    print(json_object)

    with open(output_path, "w") as outfile:
        json.dump(dict, outfile, indent=4)
