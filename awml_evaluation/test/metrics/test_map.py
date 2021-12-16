import math
from test.util.dummy_object import make_dummy_data
from typing import List
from typing import Tuple
import unittest

from awml_evaluation.common.evaluation_task import EvaluationTask
from awml_evaluation.common.label import AutowareLabel
from awml_evaluation.common.object import DynamicObject
from awml_evaluation.evaluation.metrics.metrics import MetricsScore
from awml_evaluation.evaluation.metrics.metrics_config import MetricsScoreConfig
from awml_evaluation.evaluation.result.frame_result import FrameResult
from awml_evaluation.evaluation.result.object_result import DynamicObjectWithResult
from awml_evaluation.util.debug import get_objects_with_difference


class TestMap(unittest.TestCase):
    def setUp(self):
        self.dummy_predicted_objects: List[DynamicObject] = []
        self.dummy_ground_truth_objects: List[DynamicObject] = []
        self.dummy_predicted_objects, self.dummy_ground_truth_objects = make_dummy_data()

        self.metrics_config: MetricsScoreConfig = MetricsScoreConfig(
            target_labels=[
                AutowareLabel.CAR,
                AutowareLabel.BICYCLE,
                AutowareLabel.PEDESTRIAN,
                AutowareLabel.MOTORBIKE,
            ],
            evaluation_tasks=[EvaluationTask.DETECTION],
            max_x_position_list=[100.0, 100.0, 100.0, 100.0],
            max_y_position_list=[100.0, 100.0, 100.0, 100.0],
            map_thresholds_center_distance=[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
            map_thresholds_plane_distance=[[]],
            # map_thresholds_plane_distance=[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
            map_thresholds_iou_bev=[[]],
            # map_thresholds_iou_bev=[[0.5, 0.5, 0.5, 0.5]],
            map_thresholds_iou_3d=[[]],
            # map_thresholds_iou_3d=[[0.5, 0.5, 0.5, 0.5]],
        )

    def test_map(self):
        """
        Test whether mAP and maph get correct result

        test objects:
            dummy_predicted_objects vs dummy_ground_truth_objects
        """
        ans_map: float = (1.0 + 1.0 + 0.0 + 0.0) / 4.0
        ans_maph: float = (1.0 + 1.0 + 0.0 + 0.0) / 4.0

        object_results: List[DynamicObjectWithResult] = FrameResult.get_object_results(
            predicted_objects=self.dummy_predicted_objects,
            ground_truth_objects=self.dummy_ground_truth_objects,
        )
        metrics_score: MetricsScore = MetricsScore(self.metrics_config)
        metrics_score.evaluate(
            object_results=object_results,
            ground_truth_objects=self.dummy_ground_truth_objects,
        )
        for _map in metrics_score.maps:
            self.assertAlmostEqual(_map.map, ans_map)
            self.assertAlmostEqual(_map.maph, ans_maph)

    def test_map_diff_distance(self):
        """
        Test whether mAP and maph get correct result for different distance.

        test objects:
            dummy_ground_truth_objects vs dummy_ground_truth_objects with diff_distance
        test patterns:
            Given diff_distance, check if map and maph are almost correct.
        """
        # patterns: (diff_distance, ans_map, ans_maph)
        patterns: List[Tuple[float]] = [
            # Given no diff_distance, map and maph is 1.0.
            (0.0, 1.0, 1.0),
            # Given 0.5 diff_distance for one axis, map and maph are equal to 1.0
            # since both are under the metrics threshold.
            (0.5, 1.0, 1.0),
            # Given 2.5 diff_distance for one axis, map and maph are equal to 0.0
            # since both are beyond the metrics threshold.
            (2.5, 0.0, 0.0),
        ]
        for diff_distance, ans_map, ans_maph in patterns:
            with self.subTest("diff_yaw make map"):
                diff_distance_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(diff_distance, 0.0, 0.0),
                    diff_yaw=0,
                )
                object_results: List[DynamicObjectWithResult] = FrameResult.get_object_results(
                    predicted_objects=diff_distance_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                metrics_score: MetricsScore = MetricsScore(self.metrics_config)
                metrics_score.evaluate(
                    object_results=object_results,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                for _map in metrics_score.maps:
                    self.assertAlmostEqual(_map.map, ans_map)
                    self.assertAlmostEqual(_map.maph, ans_maph)

    def test_map_diff_yaw(self):
        """
        Test whether mAP and maph get correct result for different yaw.

        test objects:
            dummy_ground_truth_objects vs dummy_ground_truth_objects with diff_yaw
        test patterns:
            Given diff_yaw, check if map and maph are almost correct.
        """
        # patterns: (diff_yaw, ans_map, ans_maph)
        patterns: List[Tuple[float]] = [
            # Given no diff_yaw, map and maph is 1.0.
            (0.0, 1.0, 1.0),
            # Given vertical diff_yaw, maph is 0.25 times map
            # since precision and recall of maph is 0.5 times those of map.
            (math.pi / 2.0, 1.0, 0.25),
            (-math.pi / 2.0, 1.0, 0.25),
            # Given opposite direction, maph is 0.0.
            (math.pi, 1.0, 0.0),
            (-math.pi, 1.0, 0.0),
            # Given diff_yaw is pi/4, maph is 0.75**2 times map
            (math.pi / 4, 1.0, 0.5625),
            (-math.pi / 4, 1.0, 0.5625),
            # Given diff_yaw is 3*pi/4, maph is 0.25**2 times map
            (3 * math.pi / 4, 1.0, 0.0625),
            (-3 * math.pi / 4, 1.0, 0.0625),
        ]
        for diff_yaw, ans_map, ans_maph in patterns:
            with self.subTest("diff_yaw make map"):
                diff_yaw_dummy_ground_truth_objects: List[
                    DynamicObject
                ] = get_objects_with_difference(
                    ground_truth_objects=self.dummy_ground_truth_objects,
                    diff_distance=(0.0, 0.0, 0.0),
                    diff_yaw=diff_yaw,
                )
                object_results: List[DynamicObjectWithResult] = FrameResult.get_object_results(
                    predicted_objects=diff_yaw_dummy_ground_truth_objects,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                metrics_score: MetricsScore = MetricsScore(self.metrics_config)
                metrics_score.evaluate(
                    object_results=object_results,
                    ground_truth_objects=self.dummy_ground_truth_objects,
                )
                for _map in metrics_score.maps:
                    self.assertAlmostEqual(_map.map, ans_map)
                    self.assertAlmostEqual(_map.maph, ans_maph)


if __name__ == "__main__":
    unittest.main()
