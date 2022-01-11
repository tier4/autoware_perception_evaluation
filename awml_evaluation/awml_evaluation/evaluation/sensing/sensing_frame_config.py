class SensingFrameConfig:
    def __init__(
        self,
        box_margin_for_0m_detection: float,
        box_margin_for_100m_detection: float,
        box_margin_for_0m_non_detection: float,
        box_margin_for_100m_non_detection: float,
    ) -> None:
        self.box_margin_for_0m_detection: float = box_margin_for_0m_detection
        self.box_margin_for_100m_detection: float = box_margin_for_100m_detection
        self.box_margin_for_0m_non_detection: float = box_margin_for_0m_non_detection
        self.box_margin_for_100m_non_detection: float = box_margin_for_100m_non_detection
