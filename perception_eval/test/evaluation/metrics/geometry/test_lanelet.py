# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from perception_eval.evaluation.metrics.geometry.lanelet import build_map_provider
from perception_eval.evaluation.metrics.geometry.lanelet import clear_lanelet_map_cache
from perception_eval.evaluation.metrics.geometry.lanelet import ExplicitMapResolver
from perception_eval.evaluation.metrics.geometry.lanelet import KNOWN_REGION_TOKENS
from perception_eval.evaluation.metrics.geometry.lanelet import LaneletMap
from perception_eval.evaluation.metrics.geometry.lanelet import LaneletMapProvider
from perception_eval.evaluation.metrics.geometry.lanelet import load_lanelet_speeds
from perception_eval.evaluation.metrics.geometry.lanelet import load_region_polygons
from perception_eval.evaluation.metrics.geometry.lanelet import T4SceneDirectoryResolver
from shapely.geometry import box
from shapely.geometry import Polygon


def _node(node_id: int, x: float, y: float) -> str:
    return f'<node id="{node_id}" lat="0" lon="0"><tag k="local_x" v="{x}"/><tag k="local_y" v="{y}"/></node>'


def _way(way_id: int, node_ids, tags=()) -> str:
    refs = "".join(f'<nd ref="{ref}"/>' for ref in node_ids)
    tag_xml = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags)
    return f'<way id="{way_id}">{refs}{tag_xml}</way>'


def _lanelet(relation_id: int, left: int, right: int, subtype: str, speed_limit=None) -> str:
    speed = f'<tag k="speed_limit" v="{speed_limit}"/>' if speed_limit is not None else ""
    return (
        f'<relation id="{relation_id}">'
        f'<member type="way" role="left" ref="{left}"/><member type="way" role="right" ref="{right}"/>'
        f'<tag k="type" v="lanelet"/><tag k="subtype" v="{subtype}"/>{speed}</relation>'
    )


# Two road lanelets: A spans x in [0, 50], y in [-5, 5] at 60 km/h; B spans x in [40, 100],
# y in [-5, 5] at 30 km/h (overlapping A on x in [40, 50]). One crosswalk_polygon area on
# x in [20, 25], y in [-8, 8]. One self-touching (bow-tie) walkway lanelet repaired by buffer(0),
# and one degenerate area way with two nodes that must be ignored.
SYNTHETIC_OSM = (
    '<?xml version="1.0"?><osm version="0.6">'
    + "".join(
        [
            _node(1, 0.0, 5.0),
            _node(2, 50.0, 5.0),
            _node(3, 0.0, -5.0),
            _node(4, 50.0, -5.0),
            _node(5, 40.0, 5.0),
            _node(6, 100.0, 5.0),
            _node(7, 40.0, -5.0),
            _node(8, 100.0, -5.0),
            _node(9, 20.0, -8.0),
            _node(10, 25.0, -8.0),
            _node(11, 25.0, 8.0),
            _node(12, 20.0, 8.0),
            # bow-tie walkway: left goes (0,10)->(10,12), right goes (0,12)->(10,10) so the ring crosses.
            _node(13, 0.0, 10.0),
            _node(14, 10.0, 12.0),
            _node(15, 0.0, 12.0),
            _node(16, 10.0, 10.0),
            _node(17, 200.0, 0.0),
            _node(18, 201.0, 0.0),
            _way(101, [1, 2]),
            _way(102, [3, 4]),
            _way(103, [5, 6]),
            _way(104, [7, 8]),
            _way(105, [9, 10, 11, 12], [("type", "crosswalk_polygon")]),
            _way(106, [13, 14]),
            _way(107, [15, 16]),
            _way(108, [17, 18], [("type", "drivable_area")]),
            _lanelet(201, 101, 102, "road", 60),
            _lanelet(202, 103, 104, "road", 30),
            _lanelet(203, 106, 107, "walkway"),
        ]
    )
    + "</osm>"
)


class TestOsmParsing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.osm_path = os.path.join(self.tmp.name, "lanelet2_map.osm")
        with open(self.osm_path, "w") as f:
            f.write(SYNTHETIC_OSM)
        clear_lanelet_map_cache()

    def tearDown(self):
        self.tmp.cleanup()

    def test_load_region_polygons(self):
        regions = load_region_polygons(self.osm_path)
        self.assertEqual(len(regions["road"]), 2)
        self.assertEqual(len(regions["crosswalk_polygon"]), 1)
        self.assertAlmostEqual(regions["crosswalk_polygon"][0].area, 80.0)
        # bow-tie repaired to a valid (possibly multi-part) polygon with positive area
        self.assertEqual(len(regions["walkway"]), 1)
        self.assertTrue(regions["walkway"][0].is_valid)
        self.assertGreater(regions["walkway"][0].area, 0.0)
        # the degenerate two-node area way is ignored
        self.assertNotIn("drivable_area", regions)
        self.assertTrue(all(polygon.is_valid for polys in regions.values() for polygon in polys))

    def test_load_lanelet_speeds_and_lowest_wins(self):
        speeds = load_lanelet_speeds(self.osm_path)
        self.assertEqual(len(speeds), 2)
        self.assertAlmostEqual(sorted(s for _, s in speeds)[0], 30.0 / 3.6)
        lanelet_map = LaneletMap.from_osm(self.osm_path)
        self.assertAlmostEqual(lanelet_map.speed_at(10.0, 0.0, 99.0), 60.0 / 3.6)
        self.assertAlmostEqual(lanelet_map.speed_at(45.0, 0.0, 99.0), 30.0 / 3.6)  # overlap -> lowest
        self.assertAlmostEqual(lanelet_map.speed_at(500.0, 0.0, 99.0), 99.0)  # off-map -> default
        self.assertEqual(LaneletMap({"road": []}).speed_at(0.0, 0.0, 7.0), 7.0)

    def test_from_osm_is_cached_per_path(self):
        first = LaneletMap.from_osm(self.osm_path)
        second = LaneletMap.from_osm(os.path.join(self.tmp.name, ".", "lanelet2_map.osm"))
        self.assertIs(first, second)


class TestLaneletMapMembership(unittest.TestCase):
    def setUp(self):
        # road x in [0, 50], y in [-10, 10]; walkway x in [0, 50], y in [10, 14] adjacent to the road.
        self.lanelet_map = LaneletMap(
            {
                "road": [Polygon([(0, -10), (50, -10), (50, 10), (0, 10)])],
                "walkway": [Polygon([(0, 10), (50, 10), (50, 14), (0, 14)])],
            }
        )

    def test_contains(self):
        xy = np.array([[10.0, 0.0], [100.0, 0.0], [10.0, 12.0]])
        np.testing.assert_array_equal(self.lanelet_map.contains(("road",), xy), [True, False, False])
        np.testing.assert_array_equal(self.lanelet_map.contains(("road", "walkway"), xy), [True, False, True])
        self.assertEqual(self.lanelet_map.contains(("road",), np.zeros((0, 2))).shape, (0,))
        # A known token absent from this map is an empty region.
        np.testing.assert_array_equal(self.lanelet_map.contains(("crosswalk",), xy), [False, False, False])

    def test_contains_margin_inward_keeps_internal_borders(self):
        # y=9.5 is 0.5 m from the road/walkway border (internal) -> still counts with a 1 m margin;
        # y=-9.5 is 0.5 m from the outer border -> dropped.
        xy = np.array([[10.0, 9.5], [10.0, -9.5], [10.0, 0.0]])
        np.testing.assert_array_equal(self.lanelet_map.contains(("road",), xy, margin=1.0), [True, False, True])

    def test_contains_margin_outward_never_claims_other_regions(self):
        # y=-10.5 is off-map within 1 m of the road -> claimed; y=10.5 is on the walkway -> not claimed.
        xy = np.array([[10.0, -10.5], [10.0, 10.5], [10.0, -12.0]])
        np.testing.assert_array_equal(
            self.lanelet_map.contains(("road",), xy, margin=1.0, expand=True), [True, False, False]
        )

    def test_intersects_footprints(self):
        footprints = [box(8.0, 8.0, 12.0, 12.0), box(8.0, 20.0, 12.0, 24.0), box(60.0, 0.0, 62.0, 1.0)]
        np.testing.assert_array_equal(self.lanelet_map.intersects(("road",), footprints), [True, False, False])
        np.testing.assert_array_equal(
            self.lanelet_map.intersects(("road",), footprints, margin=3.0), [True, False, False]
        )
        # A footprint entirely off-map but within the outward margin is claimed.
        np.testing.assert_array_equal(
            self.lanelet_map.intersects(("road",), [box(8.0, -12.0, 12.0, -10.5)], margin=2.0, expand=True), [True]
        )
        self.assertEqual(self.lanelet_map.intersects(("road",), []).shape, (0,))

    def test_region_union_and_tokens(self):
        self.assertAlmostEqual(self.lanelet_map.region_union(("road", "walkway")).area, 50 * 24)
        self.assertTrue(self.lanelet_map.region_union(("crosswalk",)).is_empty)
        self.assertEqual(set(self.lanelet_map.tokens), {"road", "walkway"})
        self.assertIn("road", KNOWN_REGION_TOKENS)
        self.assertIn("crosswalk_polygon", KNOWN_REGION_TOKENS)


class TestResolversAndProvider(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        clear_lanelet_map_cache()

    def tearDown(self):
        self.tmp.cleanup()

    def _write_map(self, directory: Path, content: str = SYNTHETIC_OSM) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "lanelet2_map.osm"
        path.write_text(content)
        return path

    def test_t4_scene_directory_resolver_flat_and_versioned(self):
        flat = self.root / "db" / "scene_flat"
        self._write_map(flat / "map")
        versioned = self.root / "db" / "scene_versioned"
        self._write_map(versioned / "0" / "map")
        no_map = self.root / "db" / "scene_without_map"
        no_map.mkdir(parents=True)

        resolver = T4SceneDirectoryResolver()
        self.assertEqual(resolver(str(flat)), flat / "map" / "lanelet2_map.osm")
        self.assertEqual(resolver(str(versioned)), versioned / "0" / "map" / "lanelet2_map.osm")
        self.assertIsNone(resolver(str(no_map)))
        self.assertIsNone(resolver(None))

        rooted = T4SceneDirectoryResolver(data_root=str(self.root))
        self.assertEqual(rooted("db/scene_flat"), flat / "map" / "lanelet2_map.osm")
        self.assertEqual(rooted(str(flat)), flat / "map" / "lanelet2_map.osm")  # absolute path also accepted
        self.assertIsNone(rooted("db/missing"))

    def test_explicit_resolver_and_build_map_provider(self):
        path = self._write_map(self.root / "maps")
        resolver = ExplicitMapResolver({"scene-a": str(path), "scene-b": str(self.root / "nope.osm")})
        self.assertEqual(resolver("scene-a"), path)
        self.assertIsNone(resolver("scene-b"))
        self.assertIsNone(resolver("scene-c"))

        provider = build_map_provider("explicit", mapping={"scene-a": str(path)})
        self.assertTrue(provider.available("scene-a"))
        self.assertFalse(provider.available("scene-b"))
        self.assertIsInstance(provider.get("scene-a"), LaneletMap)
        self.assertIs(provider.get("scene-a"), provider.get("scene-a"))
        with self.assertRaises(FileNotFoundError):
            provider.get("scene-b")
        self.assertIsInstance(build_map_provider("t4_scene_directory", data_root=str(self.root)), LaneletMapProvider)
        with self.assertRaises(ValueError):
            build_map_provider("explicit")
        with self.assertRaises(ValueError):
            build_map_provider("unknown_resolver")

    def test_corrupt_map_claims_available_but_get_raises(self):
        path = self._write_map(self.root / "corrupt", content="<osm><node id='1'")
        provider = LaneletMapProvider(ExplicitMapResolver({"scene": str(path)}))
        self.assertTrue(provider.available("scene"))
        with self.assertRaises(Exception):
            provider.get("scene")


if __name__ == "__main__":
    unittest.main()
