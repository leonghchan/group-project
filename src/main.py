#!/usr/bin/env python
"""
File: main.py
Author: Greg, Leon & Jonny
Github: https://github.com/headcase
Description: main file
"""

from load import load
from fit import analyse

def main():
    """ main function """
    
    train_data = load()
    # analyse(train_data['SalePrice'])
    analyse(train_data)

if __name__ == '__main__':
    main()


