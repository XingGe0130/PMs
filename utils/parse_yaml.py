import yaml

yaml.warnings({'YAMLLoadWarning': False})


class ParseYaml(object):

    def __init__(self, fn):
        with open(fn, "r", encoding="utf-8") as f:
            cfg = f.read()
            py_dict = yaml.load(cfg)
        self.__content = py_dict

    def get_item(self, name):
        return self.__content.get(name)
