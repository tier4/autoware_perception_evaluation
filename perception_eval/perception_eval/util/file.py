# Copyright 2022 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
from typing import Any
from typing import Tuple
import warnings

from perception_eval import __version__


def divide_file_path(full_path: str) -> Tuple[str, str, str, str, str]:
    """
    Args:
        full_path (str): './dir/subdir/filename.ext.ext2'

    Return:
        ["./dir/subdir", "filename.ext", "subdir", "filename" "ext.ext2" ]
    """

    # ./dir/subdir, filename.ext
    dir_name: str = ""
    base_name: str = ""
    dir_name, base_name = os.path.split(full_path)

    # subdir
    subdir_name: str = os.path.basename(os.path.dirname(full_path))

    # filename # .ext
    basename_without_ext: str = ""
    extension: str = ""
    basename_without_ext, extension = base_name.split(".", 1)

    return dir_name, base_name, subdir_name, basename_without_ext, extension


def load_pkl(filepath: str) -> Any:
    """Load pickle file.

    NOTE:
        Expecting serialized pickle data type is `dict`, which contains `version` information.
        ```shell
        data (dict)
            - version (str)
            - ... any data
        ```
        Returns data excluding version information.

    Args:
        filepath (str): Pickle file path.

    Returns:
        Any:
    """
    with open(filepath, "rb") as pickle_file:
        data: Any = pickle.load(pickle_file)
        if not isinstance(data, dict) or data.get("version") is None:
            warnings.warn(
                "[DEPRECATED FORMAT]: Expected serialized pkl format is `dict`, "
                f"which contains `version` information, but got type: {type(data)}, version: {data.get('version')}."
            )
            return data
        else:
            version: str = data["version"]
            if __version__.split(".")[1] != version.split(".")[1]:
                raise ValueError(
                    f"Minor version mismatch: perception_eval: {__version__}, pkl: {version}"
                )
            return data.pop("version")
