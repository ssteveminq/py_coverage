import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Polygon, Rectangle, Circle
import numpy as np
import math
from grid_map import GridMap
from grid_map_lib import test_polygon_set
from grid_based_sweep_coverage_path_planner import planning
import time
from tools import define_polygon, polygon_contains_point


test_polygon_set()
