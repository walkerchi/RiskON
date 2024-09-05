import re
import json
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import GPy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from functools import lru_cache 
from datetime import datetime
from typing import Tuple, Mapping, List, Mapping

with open("dataset/choices.json","rb") as f: 
    col2choice = json.load(f)


@lru_cache(1)
def get_df():
    df = pd.read_excel('dataset/RiskON Project - Risk events examples.xlsx')
    # df = pd.read_excel('dataset/Generated_Incident_Data_10000_Rows.xlsx')
    df["Risk Taxonomy_L1"] = df["Risk Taxonomy_L1"].apply(lambda x: re.sub(r'R[\d+\.]+ - ', '', x))
    df["Risk Taxonomy_L2"] = df["Risk Taxonomy_L2"].apply(lambda x: re.sub(r'R[\d+\.]+ - ', '', x))
    df["Risk Taxonomy_L2"] = df["Risk Taxonomy_L2"].apply(lambda x: re.sub(r'R[\d+\.]+ ', '', x))
    df["Risk Taxonomy_L3"] = df["Risk Taxonomy_L3"].apply(lambda x: re.sub(r'R[\d+\.]+ - ', '', x))
    df["Product / operation concerned"] = df["Product / operation concerned"].replace("Investment", "Investment funds")
    return df.iloc[:1000]

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
        "Client number":  [f"Client {client}"] * len(datetimes)
    })
    return df

def diff_date_viz(
    prediction:pd.DataFrame,
    y:str = "Amount in CHF",
    ):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(prediction["Date of incident"], prediction[y])


class InputEncoder:
    # intger_cols   = ["Amount in CHF"]
    # literal_cols  = ["Description of incident"]
    datetime_cols = ['Date of incident']
    integer_cols   = ["Legal entity", 
                      "Country", 
                      "Business Line", 
                      "Team", 
                      "Client number"]
    
    col_mappings = Mapping[str, Mapping[str, int]]
    inv_mappings = Mapping[str, Mapping[int, str]]
    def __init__(self):
        self.col_mappings:Mapping[str,Mapping[str,int]] = {}
        self.inv_mappings:Mapping[str,Mapping[int,str]] = {}
        for col in self.integer_cols:
            self.col_mappings[col] = {k:i for i, k in enumerate(col2choice[col])}
            self.inv_mappings[col] = {i:k for i, k in enumerate(col2choice[col])}
    def encode(self, df:pd.DataFrame)->np.ndarray:
        """
        Parameters
        ----------
        df:pd.DataFrame

        Returns
        -------
        np.ndarray [n_rows, n_features]
        """
        x = []
        for col in self.datetime_cols:
            def to_timestamp(x):
                if isinstance(x, str):
                    x = datetime.strptime(x, '%d/%m/%Y')
                return x.timestamp() / 1e9
            x.append(df[col].apply(to_timestamp).values)
        for col in self.integer_cols:
            col_mapping = self.col_mappings[col]
            x.append(df[col].map(col_mapping).values)
        x = np.stack(x, -1) 
        return x.astype(np.float32)

    def decode(self, x:np.ndarray)->pd.DataFrame:
        """
        Parameters
        ----------
        x:np.ndarray 
            [n_rows, n_features]

        Returns
        -------
        df:pd.DataFrame
        """
        df = pd.DataFrame()
        for i, col in enumerate(self.datetime_cols):
            df[col] = pd.to_datetime(x[:, i], unit='s')
        for i, col in enumerate(self.integer_cols, start=len(self.datetime_cols)):
            # breakpoint()
            inv_mapping    = self.inv_mappings[col]
            vectorized_map = np.vectorize(inv_mapping.get)
            df[col] = vectorized_map(x[:, i].astype(int))
        return df


class OutputEncoder:
    onehot_cols = [
        "Code / type of incident (E)",
        "Product / operation concerned",
        "Root_Cause_L1",
        "Root_Cause_L2",
        "Risk Taxonomy_L1",
        "Risk Taxonomy_L2",
        "Risk Taxonomy_L3",
        "Process_L1",
        "Process_L2"
    ]
    literal_cols = ["Amount in CHF"]
    def __init__(self):
        self.col_mappings:Mapping[str,Mapping[str,int]] = {}
        self.inv_mappings:Mapping[str,Mapping[int,str]] = {}
        self.n_choices:Mapping[str,int] = {}    
        for i in self.onehot_cols:
            self.col_mappings[i] = {k:i for i, k in enumerate(col2choice[i])}
            self.inv_mappings[i] = {i:k for i, k in enumerate(col2choice[i])}
            self.n_choices[i] = len(col2choice[i])
    def encode(self, df:pd.DataFrame)->np.ndarray:
        """
        Parameters
        ----------
        df: pd.DataFrame 

        Returns
        -------
        x:np.ndarray
            [n_rows, \sum n_cat * n_features ]
        """
        x = []
        for col in self.onehot_cols:
            eye = np.eye(self.n_choices[col])
            col_mapping = self.col_mappings[col]
            x.append(df[col].apply(lambda x:eye[col_mapping[x]]).values)
        x = np.concatenate([np.stack(_x,0) for _x in x], -1) 
        for col in self.literal_cols:
            x = np.concatenate([x, df[col].values[:, None]], -1)
        return x.astype(np.float32)
    
    def decode(self, x:np.ndarray)->pd.DataFrame:
        """
        Parameters
        ----------
        x:np.ndarray
            [n_rows, n_features]

        Returns
        -------
        df: pd.DataFrame
        """
        df = pd.DataFrame()
        counter = 0
        for i, col in enumerate(self.onehot_cols):
            inv_mapping = self.inv_mappings[col]
            vectorized_map = np.vectorize(inv_mapping.get)
            indices  = np.argmax(x[:, counter: counter+self.n_choices[col]], -1)
            df[col] = vectorized_map(indices)
            counter += self.n_choices[col]
        for col in self.literal_cols:
            df[col] = x[:, counter]
            counter += 1
        return df
    
class Model:
    def __init__(self):
        self.model  = None
    def fit(self, x, y):
        kernel = GPy.kern.RBF(input_dim=x.shape[-1], variance=1.0, lengthscale=1.0)
        num_inducing = 100
        self.model = GPy.models.SparseGPRegression(x, y, kernel, num_inducing=num_inducing)
        self.model.optimize(messages=True)
    def predict(self, x):
        mu,sigma = self.model.predict(x, full_cov=True)
        return mu, sigma

if __name__ == '__main__':
    input_encoder = InputEncoder()
    output_encoder = OutputEncoder()
    df = get_df()
    x = input_encoder.encode(df)
    y = output_encoder.encode(df)
    model = Model()
    model.fit(x, y)
    query = diff_date_input()
    x = input_encoder.encode(query)
    mu, std = model.predict(x)
    breakpoint()
    mu, std = output_encoder.decode(mu), output_encoder.decode(std)

    
