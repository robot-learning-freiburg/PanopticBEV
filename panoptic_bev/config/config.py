import ast
import configparser
import os

_CONVERTERS = {
    "struct": ast.literal_eval
}

_DEFAULTS_DIR = os.path.abspath(os.path.join(os.path.split(__file__)[0], "defaults"))
DEFAULTS = dict()
if os.path.exists(_DEFAULTS_DIR):
    for file in os.listdir(_DEFAULTS_DIR):
        name, ext = os.path.splitext(file)
        if ext == ".ini":
            DEFAULTS[name] = os.path.join(_DEFAULTS_DIR, file)
else:
    print("Default config not found. All parameters are taken from the provided config file.")


def load_config(config_file):
    parser = configparser.ConfigParser(allow_no_value=True, converters=_CONVERTERS)
    parser.read([config_file])
    return parser
