from enum import Enum
import pprint


def format_class_for_log(object: object) -> str:
    """[summary]
    Convert class object to str to save log

    Args:
        object (object): Class object which you want to convert for str

    Returns:
        str: str converted from class object

    Note:
        Reference is below.
        https://stackoverflow.com/questions/1036409/recursively-convert-python-object-graph-to-dictionary

    """
    return format_dict_for_log(class_to_dict(object))


def class_to_dict(object: object, class_key=None) -> dict:
    """[summary]
    Convert class object to dict

    Args:
        object (object): Class object which you want to convert to dict

    Returns:
        dict: Dict converted from class object

    Note:
        Reference is below.
        https://stackoverflow.com/questions/1036409/recursively-convert-python-object-graph-to-dictionary

    """

    if isinstance(object, dict):
        data = {}
        for (k, v) in object.items():
            data[k] = class_to_dict(v, class_key)
        return data
    elif isinstance(object, Enum):
        return str(object)
    elif hasattr(object, "_ast"):
        return class_to_dict(object._ast())
    elif hasattr(object, "__iter__") and not isinstance(object, str):
        return [class_to_dict(v, class_key) for v in object]
    elif hasattr(object, "__dict__"):
        data = dict(
            [
                (key, class_to_dict(value, class_key))
                for key, value in object.__dict__.items()
                if not callable(value) and not key.startswith("_")
            ]
        )
        if class_key is not None and hasattr(object, "__class__"):
            data[class_key] = object.__class__.__name__
        return data
    else:
        return object


def format_dict_for_log(dict: dict) -> str:
    """
    Args:
        dict (dict): dict which you want to format for logger
    Returns:
        (str) formatted str
    """
    formatted_str: str = "\n" + pprint.pformat(dict, indent=1, width=80, depth=None, compact=True)
    return formatted_str
