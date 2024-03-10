#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from streamlit_option_menu import option_menu


import json
import requests
import pandas as pd
import numpy as np


from io import StringIO
import langdetect
from langdetect import DetectorFactory, detect, detect_langs
from PIL import Image

st.set_page_config(
    page_title="MP5 Findings",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:tdi@cphbusiness.dk',
        'About': "https://docs.streamlit.io"
    }
)

st.sidebar.header("Try Me!", divider='rainbow')
# st.sidebar.success("Select a demo case from above")


banner = """
    <body style="background-color:yellow;">
            <div style="background-color:#385c7f ;padding:10px">
                <h2 style="color:white;text-align:center;">MP4 Assignment</h2>
            </div>
    </body>
    """

