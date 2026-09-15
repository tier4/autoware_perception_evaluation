# Copyright 2025 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List

from perception_eval.common.dataset import copy_frame_ground_truth
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.dataset import get_interpolated_now_frame
from perception_eval.common.dataset import get_now_frame
from perception_eval.common.schema import FrameID
from perception_eval.common.transform import HomogeneousMatrix

from test.util.dummy_object import make_dummy_data

THRESHOLD_MIN_TIME: int = 75000  # [us]
FRAME_INTERVAL: int = 100000  # [us], 10 Hz


def _make_ground_truth_frames(num_frame: int = 3) -> List[FrameGroundTruth]:
    """Make a FrameGroundTruth list which has the same objects in every frame."""
    _, dummy_ground_truth_objects = make_dummy_data()
    ego2map = HomogeneousMatrix((0, 0, 0), (1, 0, 0, 0), src=FrameID.BASE_LINK, dst=FrameID.MAP)
    return [
        FrameGroundTruth(
            unix_time=1624157578750212 + i * FRAME_INTERVAL,
            frame_name=str(i),
            objects=list(dummy_ground_truth_objects),
            transforms=[ego2map],
        )
        for i in range(num_frame)
    ]


def test_copy_frame_ground_truth() -> None:
    """Test that the copied frame does not share the object list with the source frame."""
    frame = _make_ground_truth_frames(1)[0]
    num_object = len(frame.objects)

    copied = copy_frame_ground_truth(frame)
    assert copied is not frame
    assert copied.unix_time == frame.unix_time
    assert copied.frame_name == frame.frame_name
    assert len(copied.objects) == num_object
    assert copied.objects is not frame.objects

    # filtering the copy in place must not affect the source frame
    copied.objects = []
    assert len(frame.objects) == num_object


def test_get_now_frame_does_not_expose_source_frame() -> None:
    """Test that get_now_frame() returns a copy so filtering does not break the dataset."""
    ground_truth_frames = _make_ground_truth_frames()
    num_object = len(ground_truth_frames[0].objects)

    for frame in ground_truth_frames:
        now_frame = get_now_frame(
            ground_truth_frames=ground_truth_frames,
            unix_time=frame.unix_time,
            threshold_min_time=THRESHOLD_MIN_TIME,
        )
        assert now_frame is not None
        assert now_frame.frame_name == frame.frame_name
        assert now_frame is not frame
        # simulate PerceptionEvaluationManager.filter_objects()
        now_frame.objects = []

    # calling twice must return the annotation again
    for frame in ground_truth_frames:
        assert len(frame.objects) == num_object
        now_frame = get_now_frame(
            ground_truth_frames=ground_truth_frames,
            unix_time=frame.unix_time,
            threshold_min_time=THRESHOLD_MIN_TIME,
        )
        assert len(now_frame.objects) == num_object


def test_get_now_frame_returns_none_when_too_far() -> None:
    ground_truth_frames = _make_ground_truth_frames()
    assert (
        get_now_frame(
            ground_truth_frames=ground_truth_frames,
            unix_time=ground_truth_frames[-1].unix_time + 10 * FRAME_INTERVAL,
            threshold_min_time=THRESHOLD_MIN_TIME,
        )
        is None
    )


def test_get_interpolated_now_frame_head_and_tail() -> None:
    """Test that the head/tail path of get_interpolated_now_frame() returns a copy.

    The head frame (only after frame available) and the tail frame (only before frame available)
    used to be returned by reference, so the in-place filtering of the evaluation manager emptied
    the annotation of the first/last frame of the dataset.
    """
    ground_truth_frames = _make_ground_truth_frames()
    num_object = len(ground_truth_frames[0].objects)
    tail_frame = ground_truth_frames[-1]

    # only after frame is available.
    # NOTE: a single frame dataset is used here because the frame search loop of
    # get_interpolated_now_frame() keeps overwriting `after_frame` while no before frame has been
    # found yet, so the head case is only reachable when the last frame is also the after frame.
    head_frames = _make_ground_truth_frames(1)
    head_frame = head_frames[0]
    head = get_interpolated_now_frame(
        ground_truth_frames=head_frames,
        unix_time=head_frame.unix_time - THRESHOLD_MIN_TIME // 2,
        threshold_min_time=THRESHOLD_MIN_TIME,
        return_frame_id=FrameID.MAP,
    )
    assert head is not None
    assert head is not head_frame
    head.objects = []
    assert len(head_frame.objects) == num_object

    # only before frame is available
    tail = get_interpolated_now_frame(
        ground_truth_frames=ground_truth_frames,
        unix_time=tail_frame.unix_time + THRESHOLD_MIN_TIME // 2,
        threshold_min_time=THRESHOLD_MIN_TIME,
        return_frame_id=FrameID.MAP,
    )
    assert tail is not None
    assert tail is not tail_frame
    assert len(tail.objects) == num_object
    tail.objects = []
    assert len(tail_frame.objects) == num_object

    # a second call gets the annotation back
    tail_again = get_interpolated_now_frame(
        ground_truth_frames=ground_truth_frames,
        unix_time=tail_frame.unix_time + THRESHOLD_MIN_TIME // 2,
        threshold_min_time=THRESHOLD_MIN_TIME,
        return_frame_id=FrameID.MAP,
    )
    assert len(tail_again.objects) == num_object


def test_get_interpolated_now_frame_in_between() -> None:
    """Interpolated frames were already copied, check that it stays that way."""
    ground_truth_frames = _make_ground_truth_frames()
    num_object = len(ground_truth_frames[0].objects)
    unix_time = ground_truth_frames[0].unix_time + FRAME_INTERVAL // 2

    interpolated = get_interpolated_now_frame(
        ground_truth_frames=ground_truth_frames,
        unix_time=unix_time,
        threshold_min_time=THRESHOLD_MIN_TIME,
        return_frame_id=FrameID.MAP,
    )
    assert interpolated is not None
    assert interpolated.unix_time == unix_time
    interpolated.objects = []
    for frame in ground_truth_frames:
        assert len(frame.objects) == num_object
