import yaml


class Config(dict):
    def __init__(self, filename, mode):
        super().__init__()
        config_file = "./configs/{:s}.yaml".format(filename)
        with open(config_file, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.safe_load(self._yaml)
            self._dict['MODE'] = mode

    def __getattr__(self, name):
        if name in self._dict:
            value = self._dict[name]
            if isinstance(value, dict):
                return NestedConfig(value, self)
            else:
                return value
        else:
            raise AttributeError(f"'Config' object has no attribute '{name}'")

    def print_info(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------\n')


class NestedConfig(dict):
    def __init__(self, nested_dict, config_instance):
        self.nested_dict = nested_dict
        self.config_instance = config_instance

    def __getattr__(self, name):
        if name in self.nested_dict:
            value = self.nested_dict[name]
            if isinstance(value, dict):
                return NestedConfig(value, self.config_instance)
            else:
                return value
        else:
            return self.config_instance.__getattr__(name)
