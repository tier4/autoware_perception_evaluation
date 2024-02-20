import os

import numpy as np

from simple_lanelet_loader.traffic_light_loader import TrafficLightLoader

default_map_path = "~/lanelet2_map.osm"


def refine_path(path):
    # fix ~
    path = os.path.expanduser(path)
    # fix $HOME
    path = os.path.expandvars(path)
    return path


def main(map_path: str):
    # load map obj
    map_obj = TrafficLightLoader(map_path)
    print(map_obj.get_distance_to_traffic_light_group("179563", np.array([89434.6492, 42630.8311, 5.686])))

    print(map_obj)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, default=default_map_path)

    # get args
    args = parser.parse_args()
    map_path = refine_path(args.map_path)
    main(map_path)
