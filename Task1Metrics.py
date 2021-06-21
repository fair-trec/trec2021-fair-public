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
# # Task 1 Metrics
#
# This notebook demonstrates measuring performance on Task 1.

# %% [markdown]
# ## Setup
#
# Let's load some Python modules:

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
import metrics

# %% [markdown]
# ## Load Data
#
# We will load the metadata:

# %%
pages = pd.read_json('data/trec_metadata.json.gz', lines=True)
pages.head()

# %% [markdown]
# From this metadata, we can compute the **alignment vector** for each document: the distribution over continents.

# %%
page_exp = meta[['page_id', 'geographic_locations']].explode('geographic_locations')
page_exp

# %%
page_align = page_exp.assign(x=1).pivot(index='page_id', columns='geographic_locations', values='x')
page_align = page_align.iloc[:, 1:]
page_align.fillna(0, inplace=True)
page_align

# %% [markdown]
# **TODO:** do we want to normalize pages to fractional vectors?

# %% [markdown]
# Now we need to load the topics, with their relevant document sets:

# %%
topics = pd.read_json('data/trec_topics.json.gz', lines=True)
topics.head()

# %% [markdown]
# Extract the binary qrels from relevant document lists, and re-index by ID so we can look up rels for a query:

# %%
qrels = topics[['id', 'rel_docs']].explode('rel_docs').reset_index(drop=True)
qrels.head()

# %% [markdown]
# Every document present has a relvance of 1, and absent documents are 0.
#
# Now we need the *distribution of the relevant set* for each document.

# %%
qalign = qrels.join(page_align, on='rel_docs').groupby('id').sum()
qalign

# %% [markdown]
# And normalize to sum to 1:

# %%
qa_sums = qalign.sum(axis=1)
qalign = qalign.divide(qa_sums, axis=0)
qalign

# %% [markdown]
# Now, we're going to combine these with the world population as our normlaizer. Let's create a world population vector, rather badly if we're honest:

# %%
world_pop = pd.Series([
    0.155070563,
    1.54424E-07,
    0.600202585,
    0.103663858,
    0.08609797,
    0.049616733,
    0.005348137,
], index=qalign.columns)
world_pop

# %% [markdown]
# Our target vector for each query is the arithmetic mean of its alignment vector (from the relevant set) and the global population vector.

# %%
qtarget = (qalign + world_pop) * 0.5
qtarget

# %% [markdown]
# ## Apply the Metric
#
# Let's initialize a metric:

# %%
metric = metrics.Task1Metric(qrels.set_index('id'), page_align, qtarget)

# %% [markdown]
# And load a run:

# %%
run09 = pd.read_csv('runs/task1-prec09.csv')
run09

# %% [markdown]
# Let's score each query with our metric:

# %%
met09 = run09.groupby('id')['page_id'].apply(metric).unstack()
met09.head()

# %%
met09.plot.box()

# %%
sns.relplot(x='nDCG', y='AWRF', data=met09)

# %%
