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
from scipy.stats import bootstrap
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
# Set up the RNG:

# %%
import seedbank
seedbank.initialize(20220101)
rng = seedbank.numpy_rng()

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
# And bootstrap some confidence intervals:

# %%
def boot_ci(col, name='EE-L'):
    res = bootstrap([col], statistic=np.mean, random_state=rng)
    return pd.Series({
        f'{name}.SE': res.standard_error,
        f'{name}.Lo': res.confidence_interval.low,
        f'{name}.Hi': res.confidence_interval.high,
        f'{name}.W': res.confidence_interval.high - res.confidence_interval.low
    })


# %%
run_score_ci = rank_exp.groupby('run_name')['EE-L'].apply(boot_ci).unstack()
run_score_ci

# %%
run_score_full = run_scores.join(run_score_ci)
run_score_full

# %% [markdown]
# ## Analyzing Scores
#
# What is the distribution of scores?

# %%
run_scores.describe()

# %%
sns.displot(x='EE-L', data=run_scores)
plt.savefig('figures/task1-eel-dist.pdf')
plt.show()

# %%
run_tbl_df = run_score_full[['EE-R', 'EE-D', 'EE-L']].copy()
run_tbl_df['EE-L 95% CI'] = run_score_full.apply(lambda r: "(%.3f, %.3f)" % (r['EE-L.Lo'], r['EE-L.Hi']), axis=1)
run_tbl_df

# %%
run_tbl_df.sort_values('EE-L', ascending=True, inplace=True)
run_tbl_df

# %%
run_tbl_fn = Path('figures/task2-runs.tex')
run_tbl = run_tbl_df.to_latex(float_format="%.4f", bold_rows=True, index_names=False)
run_tbl_fn.write_text(run_tbl)
print(run_tbl)

# %%
sns.relplot(x='EE-R', y='EE-D', data=run_scores)
sns.rugplot(x='EE-R', y='EE-D', data=run_scores)
plt.savefig('figures/task2-eed-eer.pdf')
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
