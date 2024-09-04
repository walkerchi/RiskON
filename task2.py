import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from tqdm import  tqdm
from sklearn.manifold import TSNE
from functools import lru_cache
from dataclasses import dataclass  
from typing import Mapping
from sentence_transformers import SentenceTransformer

with open("dataset/choices.json","rb") as f:
    col2choice = json.load(f)
    col2choice = {k:v for k,v in col2choice.items() if k in [
        "Product / operation concerned",
        "Root_Cause_L1",
        "Root_Cause_L2",
        "Risk Taxonomy_L1",
        "Risk Taxonomy_L2",
        "Risk Taxonomy_L3",
        "Process_L1",
        "Process_L2"
    ]}
    
@lru_cache(1)
def get_model():
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model

@lru_cache(1)
def get_df():
    return pd.read_excel('dataset/RiskON Project - Risk events examples.xlsx')

@lru_cache(1)
def get_inputs()->np.ndarray:
    model = get_model()
    df = get_df()
    inputs = df["Description of incident"].values
    embeddings = model.encode(inputs)
    return embeddings   

@lru_cache(1)
def get_labels()->np.ndarray:
    model = get_model()
    df = get_df()
    outputs = df[list(col2choice.keys())].values
    old_shape = outputs.shape 
    embeddings = model.encode(outputs.flatten())
    return embeddings.reshape(*old_shape, -1)

@dataclass
class TextEmb:
    texts:str # [n]
    embs:np.ndarray # [n,d]
    def __post_init__(self):
        self.text2emb = {text:emb for text, emb in zip(self.texts, self.embs)}

@lru_cache(1)
def get_choices()->Mapping[str,np.ndarray]:
    model = get_model()
    col2textemb = {}
    for col, vals in col2choice.items():
        embeddings = model.encode(vals)
        vals = np.array(vals) 
        assert vals.shape[0] == embeddings.shape[0]
        col2textemb[col] = TextEmb(texts=np.array(vals), embs=embeddings)
    return col2textemb


def cos_sim(x, y):
    x = x / (np.linalg.norm(x,axis=-1, keepdims=True) + 1e-8)
    y = y / (np.linalg.norm(y,axis=-1, keepdims=True) + 1e-8)
    return x @ y.T 


def visualize_col(col:str):
    model = get_model()
    df = get_df()
    inputs = df["Description of incident"].values
    inputs = model.encode(inputs) # [n,d]
    choices = get_choices() 
    target = choices[col].embs    # [n_label, d]
    n_inputs = inputs.shape[0]
    n_labels = target.shape[0]
    x = np.concatenate([inputs, target], axis=0)
    tsne = TSNE(n_components=2)
    x = tsne.fit_transform(x)

    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(x[:n_inputs,0], x[:n_inputs,1], color="red", label="inputs")
    ax.scatter(x[n_inputs:,0], x[n_inputs:,1], color="blue", label="labels")
    for i in range(n_labels):
        ax.text(x[n_inputs+i,0], x[n_inputs+i,1], choices[col].texts[i])
    for i in range(n_inputs):
        text = df["Description of incident"].values[i]
        if len(text) > 20:
            text = text[:20] + "..."
        ax.text(x[i,0], x[i,1], text)
    ax.legend()
    ax.set_axis_off()


    os.makedirs("outputs", exist_ok=True)
    fig.savefig(f'outputs/task2_{col.replace(" ","_").replace("/","_")}.png', dpi=400)

def visualize_all():
    for col in tqdm(col2choice.keys(), desc="Visualizing"):
        visualize_col(col)


def predict():
    inputs = get_inputs() # [n,d]
    labels = get_labels() # [n,n_label, d]
    choices = get_choices() # {col:TextEmb} * n_label

    col2text = {}
    for col, textemb in choices.items():
        target = textemb.embs # [n_label, d]
        sim    = cos_sim(inputs, target) # [n, n_label]
        prediction = sim.argmax(-1) # [n]
        col2text[f"{col}_pred"] = textemb.texts[prediction] # [n]
        col2text[col] = get_df()[col].values
    
    col2text["Description of incident"] = get_df()["Description of incident"].values

    df = pd.DataFrame.from_dict(
        col2text
    )

    os.makedirs("outputs", exist_ok=True)
    df.to_excel("putputs/task2.xlsx", index=False)
    df.to_csv("outputs/task2.csv", index=False)


if __name__ == '__main__':
    visualize_all()
    