import yaml
from importlib import import_module


class Configurator:
    """
    Class for loading configuration from yaml file
    """
    def __init__(self, config_path):
        with open(config_path) as config_file:
            self._configuration = yaml.load(config_file, Loader=yaml.FullLoader)

    def get(self, path=None, default=None):
        sub_dict = dict(self._configuration)

        if path is None:
            return sub_dict

        path_items = path.split("/")[:-1]
        data_item = path.split("/")[-1]

        try:
            for path_item in path_items:
                sub_dict = sub_dict.get(path_item)

            value = sub_dict.get(data_item, default)

            return value

        except (TypeError, AttributeError):
            return default

    @staticmethod
    def load_func(dotpath: str):
        """ load function in module.  function is right-most segment """
        module_, func = dotpath.rsplit(".", maxsplit=1)
        m = import_module(module_)

        return getattr(m, func)