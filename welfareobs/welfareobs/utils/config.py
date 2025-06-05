# -*- coding: utf-8 -*-
"""
Module Name: config.py
Description: Map JSON to something useful

Copyright (C) 2025 J.Cincotta

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
import json
import importlib
import sys
import os


class Config(object):
    def __init__(self, filename: str, add_path: str = ""):
        """
        Init
        :param filename: JSON file that is parsed on instantiation
        :param add_path: path to the welfareobs library when installed manually
        """
        self.__data = {}
        self.__filename = filename
        self.__ad_path = add_path
        with open(filename, "r") as file:
            self.__data = json.load(file)

    def __getitem__(self, src_key: str) -> any:
        """
        Get the value from the JSON dict
        :param src_key: dot notation hierarchical key
        :return: value (or resulting child dict) or key error if the key cannot be found
        """
        keys = src_key.split('.')
        value = self.__data
        for key in keys:
            if key in value:
                value = value[key]
            else:
                raise KeyError(f"Key '{key}' in {src_key} not found in the config file {self.__filename}")
        return value

    def exists(self, src_key: str) -> bool:
        """
        Confirm a key (path of keys) exists the JSON dict
        :param key: dot notation hierarchical key
        :return: bool if the key can be navigated in the JSON heirarchy
        """
        keys = src_key.split('.')
        value = self.__data
        for key in keys:
            if key in value:
                value = value[key]
            else:
                return False
        return True
    
    def as_list(self, key: str) -> [str]:
        """
        As a list (note: a single value will be converted to an array with that item as the only element.
        Other datatypes, such as a dict child node, or even a missing value, are ignored and return an empty list.
        :param key: dot notation hierarchical key
        :return: list of values
        """
        try:
            obj = self[key]
            if type(obj) is list:
                return [str(o) for o in obj]
            if type(obj) is str:
                return [obj]
        except KeyError:
            pass
        return []

    def as_string(self, key: str) -> str:
        """
        As String
        :param key: dot notation hierarchical key
        :return: string or empty string if the value can not be parsed (or the key doesn't exist)
        """
        try:
            return str(self[key])
        except KeyError:
            return ""

    def as_int(self, key: str) -> int:
        """
        As Integer
        :param key: dot notation hierarchical key
        :return: integer value or 0 if the value can not be parsed (or the key doesn't exist)
        """
        try:
            return int(self[key])
        except (KeyError, ValueError):
            return 0

    def as_float(self, key: str) -> float:
        """
        As Float
        :param key: dot notation hierarchical key
        :return: float value or 0 if the value can not be parsed (or the key doesn't exist)
        """
        try:
            return float(self[key])
        except (KeyError, ValueError):
            return 0.0

    def as_bool(self, src_key: str) -> bool:
        """
        As Boolean
        :param src_key: dot notation hierarchical key
        :return: boolean based on the value.
                 0 or "false" (case insensitive) is False
                 any other value will be interpreted as True
                 if the key does not exist, it will also return False
        """
        keys = src_key.split('.')
        value = self.__data
        for key in keys:
            if key in value:
                value = value[key]
            else:
                return False
        return str(value).lower().strip() not in ["0", "false"]

    def validate_instance(self, key: str):
        """
        Validate that the key being referenced can be 'reflected'.
        :param key: dot notation hierarchical key
        :return: True if referenced key can be 'reflected', otherwise, False
        """
        sys.path.insert(0, self.__ad_path)
        module_name, class_name = self[key].rsplit(".", 1)
        try:
            module = importlib.import_module(module_name)
            hasattr(module, class_name)
            return True
        except (KeyError, ImportError, AttributeError) as ex:
            print(ex)
            raise SyntaxError(str(ex), ex)
            return False

    def instance(self, key: str):
        """
        Reflection to get an handle to class that can be instantiated.
        :param key: dot notation hierarchical key
        :return: Return a class that can be instantiated.
                 The format for the value should be akin to "module.Class"
                 Will dynamically import the module specified to be able
                 to provide the class.
        """
        sys.path.insert(0, self.__ad_path)
        module_name, class_name = self[key].rsplit(".", 1)
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        return cls
