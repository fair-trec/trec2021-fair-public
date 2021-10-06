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
# # Task 1 Evaluation
#
# This notebook contains the evaluation for Task 1 of the TREC Fair Ranking track.

# %% [markdown]
# ## Setup
#
# We begin by loading necessary libraries:

# %%
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import binpickle

# %% [markdown]
# Set up progress bar and logging support:

# %%
from tqdm.auto import tqdm
tqdm.pandas(leave=False)

# %%
import sys, logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger('task1-eval')

# %% [markdown]
# Import metric code:

# %%
import metrics
from trecdata import scan_runs

# %% [markdown]
# And finally import the metric itself:

# %%
metric = binpickle.load('task1-eval-metric.bpk')

# %% [markdown]
# ## Importing Data

# %% [markdown]
# Let's load the runs now:

# %%
runs = pd.DataFrame.from_records(row for (task, rows) in scan_runs() if task == 1 for row in rows)
runs

# %% [markdown]
# Since we only have annotations for the first 20 for each run, limit the data:

# %%
runs = runs[runs['rank'] <= 20]

# %% [markdown]
# ## Computing Metrics
#
# We are now ready to compute the metric for each (system,topic) pair.  Let's go!

# %%
rank_awrf = runs.groupby(['run_name', 'topic_id'])['page_id'].progress_apply(metric)
rank_awrf = rank_awrf.unstack()
rank_awrf

# %% [markdown]
# Now let's average by runs:

# %%
run_scores = rank_awrf.groupby('run_name').mean()
run_scores.sort_values('Score', ascending=False)

# %% [markdown]
# ## Analyzing Scores
#
# What is the distribution of scores?

# %%
run_scores.describe()

# %%
sns.displot(x='Score', data=run_scores)
plt.show()

# %%
sns.relplot(x='nDCG', y='AWRF', data=run_scores)
sns.rugplot(x='nDCG', y='AWRF', data=run_scores)
plt.show()

# %% [markdown]
# ## Per-Topic Stats
#
# We need to return per-topic stats to each participant, at least for the score.

# %%
topic_stats = rank_awrf.groupby('topic_id').agg(['mean', 'median', 'min', 'max'])
topic_stats

# %% [markdown]
# Make final score analysis:

# %%
topic_range = topic_stats.loc[:, 'Score']
topic_range = topic_range.drop(columns=['mean'])
topic_range

# %% [markdown]
# And now we combine scores with these results to return to participants.

# %%
ret_dir = Path('results')
for system, runs in rank_awrf.groupby('run_name'):
    aug = runs.join(topic_range).reset_index().drop(columns=['run_name'])
    fn = ret_dir / f'{system}.tsv'
    log.info('writing %s', fn)
    aug.to_csv(fn, sep='\t', index=False)

# %%
