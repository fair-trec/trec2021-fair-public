# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Task 1 Metrics
#
# This notebook demonstrates measuring performance on Task 1.
#
# Before you can measure performance, you need to *prepare* the metric, by running:
#
#     python prepare-metric.py --task1
#     
# If preparing for the evaluation queries, use a the `--topics` option to specify an alternate topic file (this will not work until evaluation qrels are released).

# %% [markdown]
# ## Setup
#
# Let's load some Python modules:

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import pickle
import binpickle

# %%
from tqdm.auto import tqdm
tqdm.pandas(leave=False)

# %%
import metrics

# %% [markdown]
# ## Load Metric
#
# We will first load the metric:

# %%
t1_metric = binpickle.load('task1-train-geo-metric.bpk')

# %% [markdown]
# ## Apply the Metric
#
# Let's load a run:

# %%
run09 = pd.read_csv('runs/task1-prec09.csv')
run09

# %% [markdown]
# Let's score each query with our metric:

# %%
met09 = run09.groupby('id')['page_id'].apply(t1_metric).unstack()
met09.head()

# %%
met09.plot.box()

# %% [markdown]
# Let's plot the utility-fairness tradeoff for our topics:

# %%
sns.relplot(x='nDCG', y='AWRF', data=met09)

# %%
