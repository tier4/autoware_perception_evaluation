import logging
from typing import Dict
from typing import List
from typing import Optional
import xml.etree.ElementTree as ET  # noqa

import numpy as np


class TrafficLightLoader:
    def __init__(self, map_path) -> None:
        tree = ET.parse(map_path)
        root = tree.getroot()

        # 1. extract node, way relation from xml
        self.nodes = {}
        self.traffic_light_ways = {}
        self.traffic_light_regulatory_elements = {}

        for child in root:
            if child.tag == "node":
                ref_id, x, y, z = extract_node(child)
                self.nodes[ref_id] = [x, y, z]
            elif child.tag == "way":
                traffic_light_way = extract_traffic_light_way(child)
                if traffic_light_way:
                    self.traffic_light_ways |= traffic_light_way
            elif child.tag == "relation":
                # ids = int(child.attrib["id"])
                element = extract_traffic_light_regulatory_element(child)
                if element:
                    self.traffic_light_regulatory_elements |= element

        self.traffic_light_positions = {}
        for id, refs in self.traffic_light_ways.items():
            traffic_light_position = np.array([0.0, 0.0, 0.0])
            for ref_id in refs:
                traffic_light_position += np.array(self.nodes[ref_id])
            self.traffic_light_positions[id] = traffic_light_position / 2

    def get_distance_to_traffic_light_group(self, regulatory_element_id, origin):
        distance = 10000
        origin = np.array(origin)
        logging.info(f"regulatory_element_id: {regulatory_element_id}")
        for id, refers in self.traffic_light_regulatory_elements.items():
            if id == regulatory_element_id:
                logging.info(f"id:{id}, refers:{refers}")
                for ref_id in refers:
                    if ref_id not in self.traffic_light_positions.keys():
                        continue
                    try:
                        traffic_light_position = self.traffic_light_positions[ref_id]
                        # logging.info(f"traffic_light_position: {traffic_light_position}, {traffic_light_position-origin}")
                        dist = np.linalg.norm(traffic_light_position - origin)
                        # logging.info(f"distance={dist}")
                        if dist < distance:
                            distance = dist
                    except KeyError:
                        logging.info(f"KeyError: @{regulatory_element_id}")

        return distance


""" utils for map-loader program"""


def extract_node(child):
    x, y, z = 0, 0, 0
    for tag in child:
        if tag.attrib["k"] == "local_x":
            x = tag.attrib["v"]
        elif tag.attrib["k"] == "local_y":
            y = tag.attrib["v"]
        elif tag.attrib["k"] == "ele":
            z = tag.attrib["v"]
    return child.attrib["id"], float(x), float(y), float(z)


def extract_traffic_light_way(child):
    refs = []
    # tags = {}
    for tag in child:
        if tag.tag == "nd":
            refs.append(tag.attrib["ref"])
        # elif tag.attrib["k"] == "type":
        #     if tag.attrib["v"] == "traffic_light":
        elif tag.attrib["k"] == "subtype":
            if tag.attrib["v"] == "red_yellow_green":
                return {child.attrib["id"]: refs}


def extract_traffic_light_regulatory_element(child) -> Optional[Dict[str, List[int]]]:
    refers = []
    for tag in child:
        if tag.tag == "member":
            if tag.attrib["role"] == "refers":
                refers.append(tag.attrib["ref"])
        if tag.tag == "tag":
            if tag.attrib["k"] == "type" and tag.attrib["v"] == "regulatory_element":
                continue
            elif tag.attrib["k"] == "subtype" and tag.attrib["v"] == "traffic_light":
                continue
            else:
                return
    regulatory_element = {child.attrib["id"]: refers}
    return regulatory_element
