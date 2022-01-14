#  Copyright 2021 Rikai Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Helper functions for python reflections.
"""

import importlib


def find_class(class_name: str):
    module, cls = class_name.rsplit(".", 1)
    mod = importlib.import_module(module)
    return getattr(mod, cls)


def has_func(func_name: str) -> bool:
    """
    Assuming `x.y.z.name` as func_name,
    Check if `from x.y import z; z.name` or `from x.y.z import name` works
    """
    try:
        module, cls, func = func_name.rsplit(".", 2)
        mod = importlib.import_module(module)
        return hasattr(getattr(mod, cls), func)
    except (AttributeError, ValueError):
        try:
            module, cls = func_name.rsplit(".", 1)
            mod = importlib.import_module(module)
            return hasattr(mod, cls)
        except (ValueError, ModuleNotFoundError):
            return False
    except ModuleNotFoundError:
        return False


def find_func(func_name: str):
    """
    Assuming `x.y.z.name` as func_name
    Try `from x.y import z; z.name` first, and then `from x.y.z import name`
    """
    module, cls, func = func_name.rsplit(".", 2)
    try:
        mod = importlib.import_module(module)
        return getattr(getattr(mod, cls), func)
    except AttributeError:
        module, cls = func_name.rsplit(".", 1)
        mod = importlib.import_module(module)
        return getattr(mod, cls)
