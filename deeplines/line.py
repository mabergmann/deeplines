import math


class Line:
    def __init__(
            self,
            cx: float,
            cy: float,
            angle: float,
            length: float,
            confidence: float = 1,
    ):
        self.cx = cx
        self.cy = cy
        self.angle = angle
        self.length = length
        self.confidence = confidence

    def left(self) -> float:
        return self.cx - (self.length * abs(math.cos(self.angle)) / 2)

    def right(self) -> float:
        return self.cx + (self.length * abs(math.cos(self.angle)) / 2)

    def top(self) -> float:
        return self.cy - (self.length * abs(math.sin(self.angle)) / 2)

    def bottom(self) -> float:
        return self.cy + (self.length * abs(math.sin(self.angle)) / 2)

    def p0(self) -> tuple[float, float]:
        x = self.cx - (self.length * math.cos(self.angle) / 2)
        y = self.cy - (self.length * math.sin(self.angle) / 2)

        return x, y

    def p1(self) -> tuple[float, float]:
        x = self.cx + (self.length * math.cos(self.angle) / 2)
        y = self.cy + (self.length * math.sin(self.angle) / 2)

        return x, y
