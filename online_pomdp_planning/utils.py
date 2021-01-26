"""Utility functions that should have been provided by a standard library"""


class MovingStatistic:
    """A moving average

    Maintains average, min and max of a list of values online
    """

    def __init__(self):
        self.mean = 0.0
        self.num = 0
        self.max = float("-inf")
        self.min = float("+inf")

    def add(self, val: float):
        """Add a value to sequence that statistics are being maintained for

        :param val: the new value
        """
        self.num += 1
        self.mean = self.mean + (val - self.mean) / self.num

        self.max = max(self.max, val)
        self.min = min(self.min, val)

    def __repr__(self) -> str:
        return "MovingStatistic(mean=%s, min/max=%s/%s, n=%s)" % (
            self.mean,
            self.min,
            self.max,
            self.num,
        )
