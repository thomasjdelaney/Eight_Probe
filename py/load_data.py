import os, sys
import numpy as np
import pandas as pd
import datetime as dt
from scipy.io import loadmat

proj_dir = os.path.join(os.environ['PROJ'], 'Eight_Probe')
mat_dir = os.path.join(proj_dir, 'mat')
