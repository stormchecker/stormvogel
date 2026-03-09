"""Functionality for handling nested dictionaries easily."""

import copy
from functools import reduce
from typing import Any


def rget(d: dict, path: list) -> Any:
    """Recursively get a dict value.

    :param d: The dictionary to query.
    :param path: A list of keys forming the path to the desired value.
    :returns: The value at the given path.
    :raises KeyError: If a key along the path is not present.

    Example: ``rget(d, ['a', 'b', 'c'])`` is equivalent to ``d['a']['b']['c']``.
    """
    return reduce(lambda c, k: c.__getitem__(k), path, d)


def rset(d: dict, path: list[str], value: Any, create_new_keys: bool = False) -> dict:
    """Recursively set a dict value.

    :param d: The dictionary to modify.
    :param path: A list of keys that lead to the value to set.
    :param value: The target value.
    :param create_new_keys: If a key on the path does not exist yet, create it.

    Example: ``rset(d, ['a', 'b', 'c'], 5)`` is equivalent to ``d['a']['b']['c'] = 5``.
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
    """Merge two nested dictionaries recursively.

    Note that *dict1* is modified by reference and also returned.

    :param dict1: The base dictionary (modified in place).
    :param dict2: The dictionary to merge in. Gets priority in most cases.

    In general, *dict2* gets priority:

    - If *dict2* has a value that *dict1* does not have, the value in *dict2* is chosen.
    - If both have the same key and both are values, *dict2* is chosen.
    - If both have the same key and both are dictionaries, they are merged recursively.
    - If *dict1* has a dictionary and *dict2* has a value with the same key, *dict1* gets priority.

    Taken from StackOverflow user Anatoliy R on July 2 2024.
    https://stackoverflow.com/questions/43797333/how-to-merge-two-nested-dict-in-python
    """
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
