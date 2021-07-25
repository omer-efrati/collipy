import logging
import logging.config
import os
import pickle

import numpy as np
import pandas as pd
from scipy.constants import c, giga, centi
from collider import Collider, Data, DecayMode


logging.config.fileConfig('logging.conf')
logger = logging.getLogger('callibrationLog')

trans = (2 * giga) / (c * centi)
