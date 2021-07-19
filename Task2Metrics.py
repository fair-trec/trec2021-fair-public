# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Task 2 Metrics
#
# This notebook demonstrates measuring performance on Task 2.
#
# Before you can measure performance, you need to *prepare* the metric, by running:
#
#     python prepare-metric.py --task2
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
import pickle
import gzip

# %%
from tqdm.auto import tqdm
tqdm.pandas(leave=False)

# %%
import metrics

# %% [markdown]
# ## Load Metric
#
# We will load the compiled metric:

# %%
with gzip.open('Task2Metric.pkl.gz', 'r') as mpf:
    t2_metric = pickle.load(mpf)

# %% [markdown]
# ## Apply the Metric
#
# Let's load a run:

# %%
run09 = pd.read_csv('runs/task2-prec09.csv')
run09

# %% [markdown]
# Let's score each query with our metric:

# %%
met09 = run09.groupby('id').progress_apply(t2_metric)
met09.head()

# %%
met09.plot.box()

# %% [markdown]
# Look at the relevance-disparity relationship:

# %%
sns.relplot(x='EE-D', y='EE-R', data=met09)

# %%
