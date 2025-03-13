# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 04:37:26 
4

@author: caspe
"""

"""
The document is in three sections: 
    1. Import, styles, some hyperparameters.
    2. Loading csv & making dataframes.
    3. The dashboard
The code is a mess. I am very sorry. 
"""

import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
import dash_daq as daq
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import numpy as np


project_start_date = 2000
project_end_date = 2022

default_iso = "TUR"
default_name = "Turkey"

# =============================================================================
# STYLE SECTION.
# =============================================================================

app.run_server(port=8050, host='0.0.0.0')

