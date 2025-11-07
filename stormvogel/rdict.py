"""Functionality for handling nested dictionaries easily."""

import copy
from functools import reduce
from typing import Any


def rget(d: dict, path: list) -> Any:
    """Recursively get dict value. Throws KeyError if (nested) key not present.

    Example: rget(d, ['a', 'b', 'c']) is equivalent to d['a']['b']['c']"""
    return reduce(lambda c, k: c.__getitem__(k), path, d)


def rset(d: dict, path: list[str], value: Any, create_new_keys: bool = False) -> dict:
    """Recursively set dict value.

    Args:
        d (dict): dictionary to modify.
        path (list[str]): A list of keys that lead to the value you want to set. The first key is the first key in the path, the second key is the second key in the path, etc.
        value (Any): Target value.
        create_new_keys (bool): If a key that is on the path does not exist yet, create it. Defaults to False.

    Example: rset(d, ['a', 'b', 'c'], 5) is equivalent to d['a']['b']['c'] = 5
    """
    if len(path) == 0:
        return d

    def __rset(d: dict, path: list, value: Any):
        first = path.pop(0)
        if create_new_keys and first not in d:
            d[first] = {}
        if len(path) == 0:
            d[first] = value
        else:
            __rset(d[first], path, value)

    __rset(d, copy.deepcopy(path), value)
    return d


def merge_dict(dict1: dict, dict2: dict) -> dict:
    """Merge two nested dictionaries recursively. Note that dict1 is modified by reference and also returned.

    Args:
        dict1 (dict):
        dict2 (dict):

    In general, dict2 gets priority!
    If dict2 has a value that dict1 does not have, then the value in dict2 is chosen.
    If dict1 and dict2 have the same key, and both are VALUES, then dict2 is chosen.
    If dict1 and dict2 have the same key, and both are DICTIONARIES, then the two dictionaries are merged recursively.
    If dict1 has a DICTIONARY and dict2 has a VALUE with the same key, then dict1 gets priority.

    Taken from StackOverflow user Anatoliy R on July 2 2024.
    https://stackoverflow.com/questions/43797333/how-to-merge-two-nested-dict-in-python"""
    for key, val in dict1.items():
        if isinstance(val, dict):
            if key in dict2 and type(dict2[key] == dict):
                merge_dict(dict1[key], dict2[key])
        else:
            if key in dict2:
                dict1[key] = dict2[key]

    for key, val in dict2.items():
        if key not in dict1:
            dict1[key] = val

    return dict1
