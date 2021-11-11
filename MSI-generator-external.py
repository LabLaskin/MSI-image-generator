#====================================================================
# EXTERNAL RAW-MSI-generator SETUP
#====================================================================

#====================================================================
# Library import
#====================================================================
import os
import time
import math
from tqdm.notebook import tqdm

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# MS API
try:
    # thermo
    from pymsfilereader import MSFileReader
    # QTOF
    from multiplierz.mzAPI import mzFile
except:
    pass
#====================================================================