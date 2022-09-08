# -*- coding: utf-8 -*-
"""
Benchmark pipelines using mean absolute Disco Effect(t) as features.
"""

import pandas as pd
import numpy as np

# ML
from utils.feature_generation import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_curve,  f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# 1 - READ INPUT DATA, BINARIZE LABEL ------------
data = pd.read_excel("../data/raw/proton_binding_dataset.xlsx",
                     sheet_name='Sheet1').drop(columns="Unnamed: 0")
data['bind'] = data['AFo'].abs().apply(lambda x: 1 if x != 0 else 0)
df = data.copy()

# 2 - GENERATE FEATURES ---------------------------
df = generate_ppm_bins(df)
df = generate_categorical(df, 'polymer_name')
df['molecular_weight'] = df['polymer_name'].apply(lambda x: extract_molecular_weight(x))
zipcode_df, ppm_bin_conversion_dict, polymer_name_conversion_dict = generate_polymer_zip_codes(df, kind='cohort')
interim_modelling_data = df.copy().drop(columns=['amp_factor', 'AFo', 'SSE', 'alpha', 'beta', 'sample_size', 'proton_peak_index'])
interim_modelling_data = append_multilabel_zipcodes(interim_modelling_data, ppm_bin_conversion_dict)
interim_modelling_data = interim_modelling_data.loc[:, (interim_modelling_data != 0).any(axis=0)]
modelling_df = interim_modelling_data.copy().drop(columns=["polymer_zip_code", "concentration", 'ppm_bin', "polymer_name", "ppm_bin_codes", "polymer_name_codes"])

# take absolute disco effect, drop std dev of disco effect
modelling_df = modelling_df.abs()
modelling_df = modelling_df.drop(columns=['corr_%_attenuation0.25_std',
                                          'corr_%_attenuation0.5_std', 'corr_%_attenuation0.75_std',
                                          'corr_%_attenuation1.0_std', 'corr_%_attenuation1.25_std',
                                          'corr_%_attenuation1.5_std', 'corr_%_attenuation1.75_std'])

# set up dataframes for modelling
X = modelling_df.copy()
y = modelling_df['bind']
X = X.drop(columns=['bind'])
X.columns = X.columns.astype(str)

print(X.columns)

# 4 - Build + evaluate fully trained final model, at three random seeds -------
cv = StratifiedKFold(n_splits=5, shuffle=False)

params = [{
    'dt__max_depth': [4, 5, 6, 7, 8, 9, 10],
    'dt__min_samples_split': [2, 3, 5, 7, 10, 15, 20, 30, 40],
    'dt__min_samples_leaf':[1, 2, 3, 5, 7, 10, 15, 20]
}]


# BENCHMARK METRICS OVER 3 RANDOM SEEDS ------------
random_seed = [148, 0, 601]
fully_trained_f1_list = []
holdout_f1_list = []

# convert to numpy arrays
X = X.values
y = y.values

# Automatically select optimal number of PCs for the dataset with Minka's MLE
explore_pipe = Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components='mle'))])
explore_pipe.fit(X)
expvar_ratio = explore_pipe.named_steps['pca'].explained_variance_ratio_
n_components = len(expvar_ratio)
print("Best number of retained components from MLE is:", n_components)

best_params_list = []

for seed in random_seed:

    model = DecisionTreeClassifier(random_state=seed)

    # PCA, Modelling, GridSearch ------
    X_model_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components)),
        ('dt', model)])

    pipeCV = GridSearchCV(X_model_pipe, params, scoring='roc_auc',
                          n_jobs=-1, cv=cv, refit=True, verbose=1)
    pipeCV.fit(X=X, y=y)

    # OPTIMIZE FINAL MODEL THRESHOLD DURING SEARCH --------------------------------------------
    # https://datascience.stackexchange.com/questions/89880/classification-threshold-tuning-with-gridsearchcv
    preds = pipeCV.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, preds)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # grab the fully trained preds
    all_train_preds = preds
    all_model_preds = (pipeCV.predict_proba(X)[:, 1] >= optimal_threshold).astype(int)
    all_pred_probs = pipeCV.predict_proba(X)[:, 1]

    final_model = pipeCV.best_estimator_.named_steps['dt']

    # grab fully trained f1 -------------
    fully_trained_f1 = f1_score(y_true=y, y_pred=all_model_preds, average='macro')
    fully_trained_f1_list.append(fully_trained_f1)
    best_params_list.append(pipeCV.best_params_)

    # 5 - OUT OF SAMPLE ERROR - NESTED CROSS VALIDATION ------
    nested_pipeCV = GridSearchCV(X_model_pipe, params, scoring='roc_auc',
                                 n_jobs=-1, cv=cv, refit=True, verbose=1, return_train_score=True)

    # initiate arrays for assessment
    holdout_ypred = []
    holdout_ytrue = []

    # outer loop - each proton used as the holdout once
    loo = LeaveOneOut()

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # FIT INNER PIPELINE - SCALE, PCA, GRIDCV TO HYP TUNE AND SELECT FINAL MODEL EXCLUDING HOLDOUT
        nested_pipeCV.fit(X_train, y_train)

        # OPTIMIZE FINAL MODEL THRESHOLD on train-val data --------------------------------------------------------
        # https://datascience.stackexchange.com/questions/89880/classification-threshold-tuning-with-gridsearchcv
        preds = nested_pipeCV.predict_proba(X_train)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_train, preds)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        # TRANSFORM TEST DATA + EVALUATE FINAL MODEL ON LOOCV HOLDOUT SET ----------------------------------------------------------------
        y_pred = (nested_pipeCV.predict_proba(X_test.reshape(1, -1))
                  [:, 1] >= optimal_threshold).astype(int)

        # record holdout metrics
        holdout_ypred.append(y_pred)
        holdout_ytrue.append(y_test)

    # GRAB HOLDOUT F1 SCORE
    holdout_f1 = f1_score(y_true=np.concatenate(holdout_ytrue), y_pred=np.concatenate(holdout_ypred), average='macro')
    holdout_f1_list.append(holdout_f1)

# write results to file
f1_scores = pd.DataFrame({"random_seed": random_seed,
                              "fully_trained_f1": fully_trained_f1_list,
                              "holdout_f1": holdout_f1_list})
# add fully trained params to output
best_fully_trained_params = pd.DataFrame(best_params_list)

print(random_seed)
print(fully_trained_f1_list)
print(holdout_f1_list)
print(f1_scores)
print(best_fully_trained_params)

f1_scores.to_csv("../data/processed/mean_disco_effect_raw_f1.csv")
best_fully_trained_params.to_csv("../data/processed/meandiscoeffect_bestparams.csv")

