import json
import random
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from typing import List
from datetime import datetime
from scipy.interpolate import make_interp_spline

with open("dataset/choices.json","r") as f:    
    col2choice = json.load(f)
with open("dataset/level.json","r") as f:
    level = json.load(f)



def gen_curve(grid:int=100, n:int=5, xlims=(0,1), ylims=(0,1)):
    x_control = np.linspace(xlims[0], xlims[1], n)
    y_control = np.random.rand(n) * ylims[1] + ylims[0]
    spline = make_interp_spline(x_control, y_control, k=3)
    x = np.linspace(xlims[0], xlims[1], grid)
    return x, spline(x)

def date_range(start_date:str="2020-01-01", 
               end_date:str="2025-01-01"
               )->List[datetime]:
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')  # 'MS' stands for Month Start

    # Convert to list of datetime objects (if needed)
    datetime_list = date_range.to_pydatetime()
    return datetime_list

def diff_date_input(
    entity:int = 1,
    business:int = 1,
    team:int = 1,
    client:int = 1,
    country:str = "Luxembourg"
)->pd.DataFrame:
    assert country in ["Luxembourg", "Switzerland", "Singapore"]
    datetimes = date_range()
    df = pd.DataFrame.from_dict({
        "Date of incident": datetimes,
        "Legal entity": [f"Entity {entity}"] * len(datetimes),
        "Country": [country] * len(datetimes),
        "Business Line": [f"BL {business}"] * len(datetimes),
        "Team": [f"Team {team}"] * len(datetimes),
        "Client number":  [f"Client {client}"] * len(datetimes),
        "Amount in CHF":gen_curve(grid=len(datetimes))[1]
    })
    return df

def gen_df():
    date, = gen_curve()


    plt.plot(x, y)
    plt.show()
    pass 

if __name__ == '__main__':
    gen_df()