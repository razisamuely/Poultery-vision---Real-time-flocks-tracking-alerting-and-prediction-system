import os
import json
from recordclass import recordclass


def get_configs_as_class(configs_path):
    with open(configs_path, encoding='utf-8', errors='ignore') as json_file:
        initConfigs = json.load(json_file, object_hook=lambda d: recordclass('X', d.keys())(*d.values()))
    return initConfigs


class dirToClass(object):
    paths: None = None
    pass


class getConfigs(object):

    def get_configs_as_class(self, data=None, configs_path: str = None) -> object:
        if configs_path:
            with  open(configs_path) as f:
                data = json.load(f)

        if data:
            configs = dirToClass()
            configs.__dict__ = data
            return configs
