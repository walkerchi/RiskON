# Use of new technologies for risk events classification and analysis

## Authors
- Chi Mingyuan¹
- Petros Chatzopoulos¹
- Marco Meyer²

¹ ETH Zürich  
² University of Zurich  

---

## Introduction
During the RiskON hackathon, we were tasked to find a solution to two challenges: 
- Classify risk events using Pictet's risk taxonomy, process attribution, and root causes.
- Find the underlying risk factors of these risk events and analyze their trend and pattern.

We thus proposed two models leveraging machine learning technologies: a model for classification and another one for decomposition into primary factors and their analysis.

## What is our solution?
Our solution will contribute through cost savings by reducing the need for some employees or freeing them to perform more productive tasks.

Assuming that you receive 10 claims a day, it liberates hundreds of hours a year, which translates into a reduction in the number of employees required. The analysis model builds on this knowledge to provide insight to the risk manager and decision maker regarding their decision and risk factor loading, leading to better compliance and fewer unexpected losses.

## How do our models work?
Using the provided dataset and some simulated datasets, we developed two models to handle the task at hand.

### Classification model
This model uses the BERT transformer to encode information from the description into a high-dimensional space, which is then used to calculate the distance with the categories provided by Pictet to determine its classification. The distance is calculated using cosine similarities. In the future, the transformer can be trained on the available dataset to improve its accuracy.

### Analysis model
We again use a trained transformer on the description to identify the underlying risk factors. Using those factors, combined with other business factors in a Gaussian Process, we can infer the probability of occurrence of a risk factor conditioned on the business factor.

The advantages of these methods are their low cost. Aside from the cost of employing computer scientists to upgrade the models in the future, the cost of computing power is extremely low, at only a few dollars per day on computing platforms.

## Usage
The model currently runs on simulated dataset.

1. Install Requirements
```bash
pip install -r requiresments.txt
```

2. Run the examples for task1
```bash
python task1_example1.py
python task1_example2.py 
```

3. Run the task2
```bash
python task2.py
```

## Assumptions
The main assumptions behind our classification model, which also apply to the second model, are those of the transformer. It assumes that every natural language sentence corresponds to a high-dimensional vector in the latent space. If two sentences or words are similar in meaning, they are supposed to be close in the latent space. 

This assumption is widely accepted in computer science and linguistics.

Our analysis model is based on the assumption that the factors obey the **Gaussian Distribution**. Additionally, the relationships within factors could be modeled using an **Exponential Distribution**—if two groups of factors are far away from each other, they will affect each other less. These two distribution assumptions can be modified for customized use.

