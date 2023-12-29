from typing import Union

from .filter_param import *  # noqa
from .label_param import *  # noqa
from .metrics_param import *  # noqa

FilterParamType = Union[PerceptionFilterParam, SensingFilterParam]  # noqa
MetricsParamType = Union[PerceptionMetricsParam, SensingMetricsParam]  # noqa
