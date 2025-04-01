# -*- coding: utf-8 -*-
"""
Module Name: profile.py
Description: Profile a function by using an attribute

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


import time
from functools import wraps

def profile(func):
    """
    USAGE:

    class MyClass:
        @profile
        def long_running_method(self):
            time.sleep(2)

    obj = MyClass()
    obj.long_running_method()

    print(f"Stored execution time: {obj.long_running_method.execution_time:.4f} seconds")


    :param func: dynamically pass in the function using attribute
    :return:
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        wrapper.execution_time = end_time - start_time
        return result
    wrapper.execution_time = 0
    return wrapper
