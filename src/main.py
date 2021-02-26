#!/usr/bin/env python
"""
File: main.py
Author: Greg, Leon & Jonny
Github: https://github.com/headcase
Description: main file
"""

from load import load

def main():
    """ main function """
    
    train_data = load()
    print(train_data.head()
            )

if __name__ == '__main__':
    main()


