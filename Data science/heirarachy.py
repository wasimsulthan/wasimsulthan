import pandas as pd
import matplotlib.pyplot as plt
import sweetwiz
from AutoClean import AutoClean
from sklearn.preprocessing import MinMaxscaler
from sklearn.cluster.heirarchy  import agglomerativeclustering
from sklearn.cluster import linkage,dendogram
from Clusteval import Clusteval
import numpy as np
from sklearn.pipline import make_pipeline