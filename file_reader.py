import pandas as pd
import json
import argparse


def main():
    args = parse_args()

    lines = read_file(args.input_path)
    print(lines)

    d = lines_to_word_map(lines)
    print(d)

    write_to_json(d, args.output_path)

    return 0


def read_file(filename):
    file = open(filename, 'r')
    lines = file.read().splitlines()
    file.close()
    return lines


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
            d[cur_cat].append(words[1])
        else:
            print("Could not process: " + line)
    return d


def write_to_json(d, output_path):
    json_object = json.dumps(d, indent=4)
    print(json_object)

    with open(output_path, "w") as outfile:
        json.dump(d, outfile, indent=4)


def parse_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_path", help="path to read in keywords from")
    arg_parser.add_argument("output_path", help="path to write trend results to")
    return arg_parser.parse_args()


if __name__ == '__main__':
    main()
