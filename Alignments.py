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
# # Alignments
#
# This notebook analyzes page alignments and prepares metrics for final use.  It needs to be run to create the serialized alignment data files the metrics require.
#
# Its final output is **pickled metric objects**: an instance of the Task 1 and Task 2 metric classes, serialized to a compressed file with [binpickle][].
#
# [binpickle]: https://binpickle.lenskit.org

# %% [markdown]
# ## Setup
#
# We begin by loading necessary libraries:

# %%
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gzip
import pickle
import binpickle
from natural.size import binarysize

# %% [markdown]
# We're going to use ZStandard compression to save our metrics, so let's create a codec object:

# %%
codec = binpickle.codecs.Blosc('zstd')

# %% [markdown]
# Set up progress bar and logging support:

# %%
from tqdm.auto import tqdm
tqdm.pandas(leave=False)

# %%
import sys, logging
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
log = logging.getLogger('alignment')

# %% [markdown]
# Import metric code:

# %%
# %load_ext autoreload
# %autoreload 1

# %%
# %aimport metrics
from trecdata import scan_runs

# %% [markdown]
# ## Loading Data
#
# We first load the page metadata:

# %%
pages = pd.read_json('data/trec_metadata_eval.json.gz', lines=True)
pages = pages.drop_duplicates('page_id')
pages.info()

# %% [markdown]
# Now we will load the evaluation topics:

# %%
eval_topics = pd.read_json('data/eval-topics-with-qrels.json.gz', lines=True)
eval_topics.info()

# %%
train_topics = pd.read_json('data/trec_topics.json.gz', lines=True)
train_topics.info()

# %% [markdown]
# Train and eval topics use a disjoint set of IDs:

# %%
train_topics['id'].describe()

# %%
eval_topics['id'].describe()

# %% [markdown]
# This allows us to create a single, integrated topics list for convenience:

# %%
topics = pd.concat([train_topics, eval_topics], ignore_index=True)
topics['eval'] = False
topics.loc[topics['id'] >= 100, 'eval'] = True
topics.head()

# %% [markdown]
# Finally, a bit of hard-coded data - the world population:

# %%
world_pop = pd.Series({
    'Africa': 0.155070563,
    'Antarctica': 1.54424E-07,
    'Asia': 0.600202585,
    'Europe': 0.103663858,
    'Latin America and the Caribbean': 0.08609797,
    'Northern America': 0.049616733,
    'Oceania': 0.005348137,
})
world_pop.name = 'geography'

# %% [markdown]
# And a gender global target:

# %%
gender_tgt = pd.Series({
    'female': 0.495,
    'male': 0.495,
    'third': 0.01
})
gender_tgt.name = 'gender'
gender_tgt.sum()

# %% [markdown]
# Xarray intesectional global target:

# %%
geo_tgt_xa = xr.DataArray(world_pop, dims=['geography'])
gender_tgt_xa = xr.DataArray(gender_tgt, dims=['gender'])
int_tgt = geo_tgt_xa * gender_tgt_xa
int_tgt

# %% [markdown]
# And the order of work-needed codes:

# %%
work_order = [
    'Stub',
    'Start',
    'C',
    'B',
    'GA',
    'FA',
]

# %% [markdown]
# Now all our background data is set up.

# %% [markdown]
# ## Query Relevance
#
# We now need to get the qrels for the topics.  This is done by creating frames with entries for every relevant document; missing documents are assumed irrelevant (0).
#
# In the individual metric evaluation files, we will truncate each run to only the assessed documents (with a small amount of noise), so this is a safe way to compute.
#
# First the training topics:

# %%
train_qrels = train_topics[['id', 'rel_docs']].explode('rel_docs', ignore_index=True)
train_qrels.rename(columns={'rel_docs': 'page_id'}, inplace=True)
train_qrels['page_id'] = train_qrels['page_id'].astype('i4')
train_qrels = train_qrels.drop_duplicates()
train_qrels.head()

# %%
eval_qrels = eval_topics[['id', 'rel_docs']].explode('rel_docs', ignore_index=True)
eval_qrels.rename(columns={'rel_docs': 'page_id'}, inplace=True)
eval_qrels['page_id'] = eval_qrels['page_id'].astype('i4')
eval_qrels = eval_qrels.drop_duplicates()
eval_qrels.head()

# %% [markdown]
# And concatenate:

# %%
qrels = pd.concat([train_qrels, eval_qrels], ignore_index=True)

# %% [markdown]
# ## Page Alignments
#
# All of our metrics require page "alignments": the protected-group membership of each page.

# %% [markdown]
# ### Geography
#
# Let's start with the straight page geography alignment for the public evaluation of the training queries.  The page metadata has that; let's get the geography column.

# %%
page_geo = pages[['page_id', 'geographic_locations']].explode('geographic_locations', ignore_index=True)
page_geo.head()

# %% [markdown]
# And we will now pivot this into a matrix so we get page alignment vectors:

# %%
page_geo_align = page_geo.assign(x=1).pivot(index='page_id', columns='geographic_locations', values='x')
page_geo_align.rename(columns={np.nan: 'Unknown'}, inplace=True)
page_geo_align.fillna(0, inplace=True)
page_geo_align.head()

# %% [markdown]
# And convert this to an xarray for multidimensional usage:

# %%
page_geo_xr = xr.DataArray(page_geo_align, dims=['page', 'geography'])
page_geo_xr

# %%
binarysize(page_geo_xr.nbytes)

# %% [markdown]
# ### Gender
#
# The "undisclosed personal attribute" is gender.  Not all articles have gender as a relevant variable - articles not about a living being generally will not.
#
# We're going to follow the same approach for gender:

# %%
page_gender = pages[['page_id', 'gender']].explode('gender', ignore_index=True)
page_gender.fillna('unknown', inplace=True)
page_gender.head()

# %% [markdown]
# We need to do a little targeted repair - there is an erroneous record of a gender of "Taira no Kiyomori" is actually male. Replace that:

# %%
page_gender = page_gender.loc[page_gender['gender'] != 'Taira no Kiyomori']

# %% [markdown]
# Now, we're going to do a little more work to reduce the dimensionality of the space.  Points:
#
# 1. Trans men are men
# 2. Trans women are women
# 3. Cisgender is an adjective that can be dropped for the present purposes
#
# The result is that we will collapse "transgender female" and "cisgender female" into "female".
#
# The **downside** to this is that trans men are probabily significantly under-represented, but are now being collapsed into the dominant group.

# %%
pgcol = page_gender['gender']
pgcol = pgcol.str.replace(r'(?:tran|ci)sgender\s+((?:fe)?male)', r'\1', regex=True)

# %% [markdown]
# Now, we're going to group the remaining gender identities together under the label 'third'.  As noted above, this is a debatable exercise that collapses a lot of identity.

# %%
genders = ['unknown', 'male', 'female', 'third']
pgcol[~pgcol.isin(genders)] = 'third'

# %% [markdown]
# Now put this column back in the frame and deduplicate.

# %%
page_gender['gender'] = pgcol
page_gender = page_gender.drop_duplicates()

# %% [markdown]
# And make an alignment matrix (reordering so 'unknown' is first for consistency):

# %%
page_gend_align = page_gender.assign(x=1).pivot(index='page_id', columns='gender', values='x')
page_gend_align.fillna(0, inplace=True)
page_gend_align = page_gend_align.reindex(columns=['unknown', 'female', 'male', 'third'])
page_gend_align.head()

# %% [markdown]
# Let's see how frequent each of the genders is:

# %%
page_gend_align.sum(axis=0).sort_values(ascending=False)

# %% [markdown]
# And convert to an xarray:

# %%
page_gend_xr = xr.DataArray(page_gend_align, dims=['page', 'gender'])
page_gend_xr

# %%
binarysize(page_gend_xr.nbytes)

# %% [markdown]
# ### Intersectional Alignment
#
# We'll now convert this data array to an **intersectional** alignment array:

# %%
page_xalign = page_geo_xr * page_gend_xr
page_xalign

# %%
binarysize(page_xalign.nbytes)

# %% [markdown]
# Make sure that did the right thing and we have intersectional numbers:

# %%
page_xalign.sum(axis=0)

# %% [markdown]
# And make sure combination with targets work as expected:

# %%
(page_xalign.sum(axis=0) + int_tgt) * 0.5

# %% [markdown]
# ## Task 1 Metric Preparation
#
# Now that we have our alignments and qrels, we are ready to prepare the Task 1 metrics.
#
# Task 1 ignores the "unknown" alignment category, so we're going to create a `kga` frame (for **K**nown **G**eographic **A**lignment), and corresponding frames for intersectional alignment.

# %%
page_kga = page_geo_align.iloc[:, 1:]
page_kga.head()

# %% [markdown]
# Intersectional is a little harder to do, because things can be **intersectionally unknown**: we may know gender but not geography, or vice versa.  To deal with these missing values for Task 1, we're going to ignore *totally unknown* values, but keep partially-known as a category.
#
# We also need to ravel our tensors into a matrix for compatibility with the metric code. Since 'unknown' is the first value on each axis, we can ravel, and then drop the first column.

# %%
xshp = page_xalign.shape
xshp = (xshp[0], xshp[1] * xshp[2])
page_xa_df = pd.DataFrame(page_xalign.values.reshape(xshp), index=page_xalign.indexes['page'])
page_xa_df.head()

# %% [markdown]
# And drop unknown, to get our page alignment vectors:

# %%
page_kia = page_xa_df.iloc[:, 1:]

# %% [markdown]
# ### Geographic Alignment
#
# We'll start with the metric configuration for public training data, considering only geographic alignment.  We configure the metric to do this for both the training and the eval queries.
#
# #### Training Queries

# %%
train_qalign = train_qrels.join(page_kga, on='page_id').drop(columns=['page_id']).groupby('id').sum()
tqa_sums = train_qalign.sum(axis=1)
train_qalign = train_qalign.divide(tqa_sums, axis=0)

# %%
train_qalign.head()

# %%
train_qtarget = (train_qalign + world_pop) * 0.5
train_qtarget.head()

# %% [markdown]
# And we can prepare a metric and save it:

# %%
t1_train_metric = metrics.Task1Metric(train_qrels.set_index('id'), page_kga, train_qtarget)
binpickle.dump(t1_train_metric, 'task1-train-geo-metric.bpk', codec=codec)

# %% [markdown]
# #### Eval Queries
#
# Do the same thing for the eval data for a geo-only eval metric:

# %%
eval_qalign = eval_qrels.join(page_kga, on='page_id').drop(columns=['page_id']).groupby('id').sum()
eqa_sums = eval_qalign.sum(axis=1)
eval_qalign = eval_qalign.divide(eqa_sums, axis=0)
eval_qtarget = (eval_qalign + world_pop) * 0.5
t1_eval_metric = metrics.Task1Metric(eval_qrels.set_index('id'), page_kga, eval_qtarget)
binpickle.dump(t1_eval_metric, 'task1-eval-geo-metric.bpk', codec=codec)

# %% [markdown]
# ### Intersectional Alignment
#
# Now we need to apply similar logic, but for the intersectional (geography * gender) alignment.
#
# As noted as above, we need to carefully handle the unknown cases.

# %% [markdown]
# #### Demo
#
# To demonstrate how the logic works, let's first work it out in cells for one query (1).
#
# What are its documents?

# %%
qdf = qrels[qrels['id'] == 1]
qdf.name = 1
qdf

# %% [markdown]
# We can use these page IDs to get its alignments:

# %%
q_xa = page_xalign.loc[qdf['page_id'].values, :, :]
q_xa

# %% [markdown]
# Summing over the first axis ('page') will produce an alignment matrix:

# %%
q_am = q_xa.sum(axis=0)
q_am

# %% [markdown]
# Now we need to do reset the (0,0) coordinate (full unknown), and normalize to a proportion.

# %%
q_am[0, 0] = 0
q_am = q_am / q_am.sum()
q_am

# %% [markdown]
# Ok, now we have to - very carefully - average with our target modifier.  There are three groups:
#
# - known (use intersectional target)
# - known-geo (use geo target)
# - known-gender (use gender target)
#
# For each of these, we need to respect the fraction of the total it represents.  Let's compute those fractions:

# %%
q_fk_all = q_am[1:, 1:].sum()
q_fk_geo = q_am[1:, :1].sum()
q_fk_gen = q_am[:1, 1:].sum()
q_fk_all, q_fk_geo, q_fk_gen

# %% [markdown]
# And now do some surgery.  Weighted-average to incorporate the target for fully-known:

# %%
q_tm = q_am.copy()
q_tm[1:, 1:] *= 0.5
q_tm[1:, 1:] += int_tgt * 0.5 * q_fk_all
q_tm

# %% [markdown]
# And for known-geo:

# %%
q_tm[1:, :1] *= 0.5
q_tm[1:, :1] += geo_tgt_xa * 0.5 * q_fk_geo

# %% [markdown]
# And known-gender:

# %%
q_tm[:1, 1:] *= 0.5
q_tm[:1, 1:] += gender_tgt_xa * 0.5 * q_fk_gen

# %%
q_tm

# %% [markdown]
# Now we can unravel this and drop the first entry:

# %%
q_tm.values.ravel()[1:]


# %% [markdown]
# #### Implementation
#
# Now, to do this for every query, we'll use a function that takes a data frame for a query's relevant docs and performs all of the above operations:

# %%
def query_xalign(qdf):
    pages = qdf['page_id']
    pages = pages[pages.isin(page_xalign.indexes['page'])]
    q_xa = page_xalign.loc[pages.values, :, :]
    q_am = q_xa.sum(axis=0)

    # clear and normalize
    q_am[0, 0] = 0
    q_am = q_am / q_am.sum()
    
    # compute fractions in each section
    q_fk_all = q_am[1:, 1:].sum()
    q_fk_geo = q_am[1:, :1].sum()
    q_fk_gen = q_am[:1, 1:].sum()
    
    # known average
    q_am[1:, 1:] *= 0.5
    q_am[1:, 1:] += int_tgt * 0.5 * q_fk_all
    
    # known-geo average
    q_am[1:, :1] *= 0.5
    q_am[1:, :1] += geo_tgt_xa * 0.5 * q_fk_geo
    
    # known-gender average
    q_am[:1, 1:] *= 0.5
    q_am[:1, 1:] += gender_tgt_xa * 0.5 * q_fk_gen
    
    # and return the result
    return pd.Series(q_am.values.ravel()[1:])


# %%
query_xalign(qdf)

# %% [markdown]
# Now with that function, we can compute the alignment vector for each query.

# %%
train_qtarget = train_qrels.groupby('id').apply(query_xalign)
train_qtarget

# %% [markdown]
# And save:

# %%
t1_train_metric = metrics.Task1Metric(train_qrels.set_index('id'), page_kia, train_qtarget)
binpickle.dump(t1_train_metric, 'task1-train-metric.bpk', codec=codec)

# %% [markdown]
# Do the same for eval:

# %%
eval_qtarget = eval_qrels.groupby('id').apply(query_xalign)
t1_eval_metric = metrics.Task1Metric(eval_qrels.set_index('id'), page_kia, eval_qtarget)
binpickle.dump(t1_eval_metric, 'task1-eval-metric.bpk', codec=codec)

# %% [markdown]
# ## Task 2 Metric Preparation
#
# Task 2 requires some different preparation.
#
# We're going to start by computing work-needed information:

# %%
page_work = pages.set_index('page_id').quality_score_disc.astype(pd.CategoricalDtype(ordered=True))
page_work = page_work.cat.reorder_categories(work_order)
page_work.name = 'quality'

# %% [markdown]
# ### Work and Target Exposure
#
# The first thing we need to do to prepare the metric is to compute the work-needed for each topic's pages, and use that to compute the target exposure for each (relevant) page in the topic.
#
# This is because an ideal ranking orders relevant documents in decreasing order of work needed, followed by irrelevant documents.  All relevant documents at a given work level should receive the same expected exposure.
#
# First, look up the work for each query page ('query page work', or qpw):

# %%
qpw = qrels.join(page_work, on='page_id')
qpw

# %% [markdown]
# And now  use that to compute the number of documents at each work level:

# %%
qwork = qpw.groupby(['id', 'quality'])['page_id'].count()
qwork


# %% [markdown]
# Now we need to convert this into target exposure levels.  This function will, given a series of counts for each work level, compute the expected exposure a page at that work level should receive.

# %%
def qw_tgt_exposure(qw_counts: pd.Series) -> pd.Series:
    if 'id' == qw_counts.index.names[0]:
        qw_counts = qw_counts.reset_index(level='id', drop=True)
    qwc = qw_counts.reindex(work_order, fill_value=0).astype('i4')
    tot = int(qwc.sum())
    da = metrics.discount(tot)
    qwp = qwc.shift(1, fill_value=0)
    qwc_s = qwc.cumsum()
    qwp_s = qwp.cumsum()
    res = pd.Series(
        [np.mean(da[s:e]) for (s, e) in zip(qwp_s, qwc_s)],
        index=qwc.index
    )
    return res


# %% [markdown]
# We'll then apply this to each topic, to determine the per-topic target exposures:

# %%
qw_pp_target = qwork.groupby('id').apply(qw_tgt_exposure)
qw_pp_target.name = 'tgt_exposure'
qw_pp_target

# %% [markdown]
# We can now merge the relevant document work categories with this exposure, to compute the target exposure for each relevant document:

# %%
qp_exp = qpw.join(qw_pp_target, on=['id', 'quality'])
qp_exp = qp_exp.set_index(['id', 'page_id'])['tgt_exposure']
qp_exp.index.names = ['q_id', 'page_id']
qp_exp

# %% [markdown]
# ### Geographic Alignment
#
# Now that we've computed per-page target exposure, we're ready to set up the geographic alignment vectors for computing the per-*group* expected exposure with geographic data.
#
# We're going to start by getting the alignments for relevant documents for each topic:

# %%
qp_geo_align = qrels.join(page_geo_align, on='page_id').set_index(['id', 'page_id'])
qp_geo_align.index.names = ['q_id', 'page_id']
qp_geo_align

# %% [markdown]
# Now we need to compute the per-query target exposures.  This starst with aligning our vectors:

# %%
qp_geo_exp, qp_geo_align = qp_exp.align(qp_geo_align, fill_value=0)

# %% [markdown]
# And now we can multiply the exposure vector by the alignment vector, and summing by topic - this is equivalent to the matrix-vector multiplication on a topic-by-topic basis.

# %%
qp_aexp = qp_geo_align.multiply(qp_geo_exp, axis=0)
q_geo_align = qp_aexp.groupby('q_id').sum()

# %% [markdown]
# Now things get a *little* weird.  We want to average the empirical distribution with the world population to compute our fairness target.  However, we don't have empirical data on the distribution of articles that do or do not have geographic alignments.
#
# Therefore, we are going to average only the *known-geography* vector with the world population.  This proceeds in N steps:
#
# 1. Normalize the known-geography matrix so its rows sum to 1.
# 2. Average each row with the world population.
# 3. De-normalize the known-geography matrix so it is in the original scale, but adjusted w/ world population
# 4. Normalize the *entire* matrix so its rows sum to 1
#
# Let's go.

# %%
qg_known = q_geo_align.drop(columns=['Unknown'])

# %% [markdown]
# Normalize (adding a small value to avoid division by zero - affected entries will have a zero numerator anyway):

# %%
qg_ksums = qg_known.sum(axis=1)
qg_kd = qg_known.divide(np.maximum(qg_ksums, 1.0e-6), axis=0)

# %% [markdown]
# Average:

# %%
qg_kd = (qg_kd + world_pop) * 0.5

# %% [markdown]
# De-normalize:

# %%
qg_known = qg_kd.multiply(qg_ksums, axis=0)

# %% [markdown]
# Recombine with the Unknown column:

# %%
q_geo_tgt = q_geo_align[['Unknown']].join(qg_known)

# %% [markdown]
# Normalize targets:

# %%
q_geo_tgt = q_geo_tgt.divide(q_geo_tgt.sum(axis=1), axis=0)
q_geo_tgt

# %% [markdown]
# This is our group exposure target distributions for each query, for the geographic data.  We're now ready to set up the matrix.

# %%
train_geo_qtgt = q_geo_tgt.loc[train_topics['id']]
eval_geo_qtgt = q_geo_tgt.loc[eval_topics['id']]

# %%
t2_train_geo_metric = metrics.Task2Metric(train_qrels.set_index('id'), 
                                          page_geo_align, page_work, 
                                          train_geo_qtgt)
binpickle.dump(t2_train_geo_metric, 'task2-train-geo-metric.bpk', codec=codec)

# %%
t2_eval_geo_metric = metrics.Task2Metric(eval_qrels.set_index('id'), 
                                         page_geo_align, page_work, 
                                         eval_geo_qtgt)
binpickle.dump(t2_eval_geo_metric, 'task2-eval-geo-metric.bpk', codec=codec)


# %% [markdown]
# ### Intersectional Alignment
#
# Now we need to compute the intersectional targets for Task 2.  We're going to take a slightly different approach here, based on the intersectional logic for Task 1, because we've come up with better ways to write the code, but the effect is the same: only known aspects are averaged.
#
# We'll write a function very similar to the one for Task 1:

# %%
def query_xideal(qdf, ravel=True):
    pages = qdf['page_id']
    pages = pages[pages.isin(page_xalign.indexes['page'])]
    q_xa = page_xalign.loc[pages.values, :, :]
    
    # now we need to get the exposure for the pages, and multiply
    p_exp = qp_exp.loc[qdf.name]
    assert p_exp.index.is_unique
    p_exp = xr.DataArray(p_exp, dims=['page'])
    
    # and we multiply!
    q_xa = q_xa * p_exp

    # normalize into a matrix (this time we don't clear)
    q_am = q_xa.sum(axis=0)
    q_am = q_am / q_am.sum()
    
    # compute fractions in each section - combined with q_am[0,0], this should be about 1
    q_fk_all = q_am[1:, 1:].sum()
    q_fk_geo = q_am[1:, :1].sum()
    q_fk_gen = q_am[:1, 1:].sum()
    
    # known average
    q_am[1:, 1:] *= 0.5
    q_am[1:, 1:] += int_tgt * 0.5 * q_fk_all
    
    # known-geo average
    q_am[1:, :1] *= 0.5
    q_am[1:, :1] += geo_tgt_xa * 0.5 * q_fk_geo
    
    # known-gender average
    q_am[:1, 1:] *= 0.5
    q_am[:1, 1:] += gender_tgt_xa * 0.5 * q_fk_gen
    
    # and return the result
    if ravel:
        return pd.Series(q_am.values.ravel())
    else:
        return q_am


# %% [markdown]
# Test this function out:

# %%
query_xideal(qdf, ravel=False)

# %% [markdown]
# And let's go!

# %%
q_xtgt = qrels.groupby('id').progress_apply(query_xideal)
q_xtgt

# %%
train_qtgt = q_xtgt.loc[train_topics['id']]
eval_qtgt = q_xtgt.loc[eval_topics['id']]

# %%
t2_train_metric = metrics.Task2Metric(train_qrels.set_index('id'), 
                                      page_xa_df, page_work, 
                                      train_qtgt)
binpickle.dump(t2_train_metric, 'task2-train-metric.bpk', codec=codec)

# %%
t2_eval_metric = metrics.Task2Metric(eval_qrels.set_index('id'), 
                                     page_xa_df, page_work, 
                                     eval_qtgt)
binpickle.dump(t2_eval_metric, 'task2-eval-metric.bpk', codec=codec)

# %%
