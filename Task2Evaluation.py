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
# # Task 2 Evaluation
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
metric = binpickle.load('task2-eval-metric.bpk')

# %% [markdown]
# ## Importing Data
#
#

# %% [markdown]
# Let's load the runs now:

# %%
runs = pd.DataFrame.from_records(row for (task, rows) in scan_runs() if task == 2 for row in rows)
runs

# %%
runs.head()

# %% [markdown]
# We also need to load our topic eval data:

# %%
topics = pd.read_json('data/eval-topics-with-qrels.json.gz', lines=True)
topics.head()

# %% [markdown]
# Tier 2 is the top 5 docs of the first 25 rankings.  Further, we didn't complete Tier 2 for all topics.

# %%
t2_topics = topics.loc[topics['max_tier'] >= 2, 'id']

# %%
r_top5 = runs['rank'] <= 5
r_first25 = runs['seq_no'] <= 25
r_done = runs['topic_id'].isin(t2_topics)
runs = runs[r_done & r_top5 & r_first25]
runs.info()

# %% [markdown]
# ## Computing Metrics
#
# We are now ready to compute the metric for each (system,topic) pair.  Let's go!

# %%
rank_exp = runs.groupby(['run_name', 'topic_id']).progress_apply(metric)
# rank_exp = rank_awrf.unstack()
rank_exp

# %% [markdown]
# Now let's average by runs:

# %%
run_scores = rank_exp.groupby('run_name').mean()
run_scores

# %% [markdown]
# ## Analyzing Scores
#
# What is the distribution of scores?

# %%
run_scores.describe()

# %%
sns.displot(x='EE-L', data=run_scores)
plt.show()

# %%
run_scores.sort_values('EE-L', ascending=False)

# %%
sns.relplot(x='EE-D', y='EE-R', data=run_scores)
sns.rugplot(x='EE-D', y='EE-R', data=run_scores)
plt.show()

# %% [markdown]
# ## Per-Topic Stats
#
# We need to return per-topic stats to each participant, at least for the score.

# %%
topic_stats = rank_exp.groupby('topic_id').agg(['mean', 'median', 'min', 'max'])
topic_stats

# %% [markdown]
# Make final score analysis:

# %%
topic_range = topic_stats.loc[:, 'EE-L']
topic_range = topic_range.drop(columns=['mean'])
topic_range

# %% [markdown]
# And now we combine scores with these results to return to participants.

# %%
ret_dir = Path('results')
for system, runs in rank_exp.groupby('run_name'):
    aug = runs.join(topic_range).reset_index().drop(columns=['run_name'])
    fn = ret_dir / f'{system}.tsv'
    log.info('writing %s', fn)
    aug.to_csv(fn, sep='\t', index=False)

# %%
