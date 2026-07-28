# Copyright 2026 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Resolve a 2D traffic-light `object_ann.instance_token` to a Regulatory Element (RE) ID.

`DynamicObject2D.uuid` for a traffic-light annotation is the RE ID. The only convention
this package assumes is the one already baked into standard T4 datasets:
`Instance.instance_name` ending in `"...:<RE ID>"` (see `LegacyInstanceNameResolver`).

Any other instance-to-RE relation source (a separate relation table, a map lookup, etc.)
is not part of any officially agreed t4dataset IF today, so this module does not assume
one exists, read one, or parse a map itself. A caller that has such a source available
can supply its own `TrafficLightIdResolver` via
`perception_eval.common.dataset.load_all_datasets(..., traffic_light_id_resolver_factory=...)`
instead of forking this file.
"""
from __future__ import annotations

from typing import Any
from typing import Dict
from typing import List
from typing import Protocol
from typing import runtime_checkable

__all__ = [
    "LegacyInstanceNameResolver",
    "TrafficLightIdResolver",
    "TrafficLightRelationError",
]


class TrafficLightRelationError(Exception):
    """Raised when a traffic-light `object_ann.instance_token` cannot be resolved to an
    RE ID."""


@runtime_checkable
class TrafficLightIdResolver(Protocol):
    """Resolves a 2D traffic-light `object_ann.instance_token` to a Regulatory Element ID.

    Implementations are constructed once per dataset load (see
    `perception_eval.common.dataset._load_dataset`) and reused across every
    frame/annotation of that dataset; `resolve_re_id` itself must not re-scan any table.
    """

    def resolve_re_id(self, instance_token: str) -> str:
        """Return the Regulatory Element ID related to `instance_token`.

        Raises:
            TrafficLightRelationError: If `instance_token` cannot be resolved.
        """
        ...


class LegacyInstanceNameResolver:
    """Resolves the RE ID from `Instance.instance_name` (`"...:<RE ID>"`).

    The RE ID is the text after the last `:` in `instance_name`. This is the only
    instance-to-RE convention perception_eval assumes by default.
    """

    def __init__(self, instance_records: List[Dict[str, Any]]) -> None:
        """
        Args:
            instance_records (List[Dict[str, Any]]): Raw `instance.json` records
                (e.g. `nusc.instance`).
        """
        self._instance_by_token: Dict[str, Dict[str, Any]] = {record["token"]: record for record in instance_records}

    def resolve_re_id(self, instance_token: str) -> str:
        record = self._instance_by_token.get(instance_token)
        if record is None:
            raise TrafficLightRelationError(f"instance_token '{instance_token}' is not found in instance.json.")

        instance_name: str = record.get("instance_name", "")
        if ":" not in instance_name:
            raise TrafficLightRelationError(
                "Legacy TLR instance_name must contain ':' followed by the Regulatory "
                f"Element ID, but got instance_name={instance_name!r} for "
                f"instance_token={instance_token!r}."
            )

        re_id = instance_name.rsplit(":", 1)[-1]
        if re_id == "":
            raise TrafficLightRelationError(
                "Legacy TLR instance_name has an empty Regulatory Element ID: "
                f"instance_name={instance_name!r} (instance_token={instance_token!r})."
            )
        return re_id
