import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pytest
from pm25aqi.config import CATEGORIES
from pm25aqi.data import categories_from_aqi

@pytest.fixture
def toy():
    rng=np.random.default_rng(123)
    n=144
    patterns=np.arange(n)//2
    x=pd.DataFrame({
       'CO AQI Value':(patterns*3 % 60).astype(float),
       'Ozone AQI Value':(patterns*5 % 50).astype(float),
       'NO2 AQI Value':(patterns*7 % 40).astype(float),
       'Country':np.where(patterns%3==0,'A',np.where(patterns%3==1,'B','C'))})
    y=(x['CO AQI Value']+x['Ozone AQI Value']*.5+rng.uniform(0,3,n)).clip(0,500)
    x['PM2.5 AQI Value']=y
    x['PM2.5 AQI Category']=categories_from_aqi(y)
    x['AQI Value']=99999
    x['AQI Category']='TARGET_BASED_FAKE'
    return x
