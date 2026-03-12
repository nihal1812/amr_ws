import math
from typing import Tuple

def world_to_grid(x: float, y: float, origin_x: float, origin_y: float, res: float) -> Tuple[int,int]:
    i = int(math.floor((x - origin_x) / res))
    j = int(math.floor((y - origin_y) / res))
    return i, j

def grid_to_world(i: int, j: int, origin_x: float, origin_y: float, res: float) -> Tuple[float,float]:
    x = origin_x + (i + 0.5) * res
    y = origin_y + (j + 0.5) * res
    return x, y
