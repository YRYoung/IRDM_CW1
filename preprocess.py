#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import re

words = r'?:\w+[\']?\w'
numbers = r'?:[-]?\d+(?:[,-.]\d+'
symbols = r'?:[\\(){}[\]=&|^+<>/*%;\.,"\'?!~——]'

token_pattern = fr'({symbols})|({numbers})*|({words}))'


def get_tokens(text) -> list:
    print('tokenizing')
    tokens = re.findall(token_pattern, string=text)

    return tokens


def read_txt(filename='./data/passage-collection.txt'):
    print('reading txt')
    with open(filename, "r", encoding='utf-8') as f:  # 打开文件
        data = f.read()  # 读取文件
    return data


def preprocessing():
    data = read_txt()
    words = get_tokens(data)
    return delnum(words)



remove_list = '\\(){}[\]=&|^+<>/*%;.,\"\'?!~——'


def delnum(word_list):
    b = filter(lambda x: x not in remove_list, word_list)
    return list(b)


if __name__ == '__main__':
    preprocessing()
