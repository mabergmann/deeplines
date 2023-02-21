class Line(object):
    def __init__(
            self,
            cx: float,
            cy: float,
            angle: float,
            length: float,
            confidence: float = 1
    ):
        self.cx = cx
        self.cy = cy
        self.angle = angle
        self.length = length
        self.confidence = confidence
