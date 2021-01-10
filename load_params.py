import sys

import re
import numpy as np
import pandas as pd
import pickle

def load_data(data_filepath):
    with open(data_filepath, 'rb') as f:
        return pickle.load(f)

def main():
    if len(sys.argv) == 2:
        data_filepath = sys.argv[1]
        print('Loading data...\n    DATABASE: {}'.format(data_filepath))
        
        data = load_data(data_filepath)
        print('Best parameters: \n', data)
    else:
        print('Please provide the filepath of the of the pickle file'\
              '\nExample: python '\
              'load_params.py ../data/best_params.pkl')

if __name__ == '__main__':
    main()