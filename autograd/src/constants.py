from enum import StrEnum


class AutogradBackends(StrEnum):
    """ PY backend is for prototyping """
    
    PYTHON = "py"
    C = "c"
