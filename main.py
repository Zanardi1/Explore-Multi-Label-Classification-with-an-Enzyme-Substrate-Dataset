import collections
import gc
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.simplefilter('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
import lightgbm as lgb


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def generate_features(train, test, cat_cols, num_cols):
    features = cat_cols + num_cols
    df = pd.concat([train, test], axis=0, copy=False)
    for c in cat_cols + num_cols:
        df[f'count_{c}'] = df.groupby(c)[c].transform('count')
        features.append(f'count_{c}')
    for c in cat_cols:
        for n in num_cols:
            df[f'mean_{n}_per_{c}'] = df.groupby(c)[n].transform('mean')
    return df.iloc[:len(train), :], df.iloc[len(train):, :], features


def convert_to_cat(value):
    if value in dic:
        return dic[value]
    else:
        return 9999


def calc_log_loss_weight(y_true):
    nc = np.bincount(y_true)
    w0, w1 = 1 / (nc[0] / y_true.shape[0]), 1 / (nc[1] / y_true.shape[0])
    return w0, w1


def lightgbm_training(x_train: pd.DataFrame, y_train: pd.DataFrame, x_valid: pd.DataFrame, y_valid: pd.DataFrame,
                      features: list, categorical_features: list):
    train_w0, train_w1 = calc_log_loss_weight(y_train)
    valid_w0, valid_w1 = calc_log_loss_weight(y_valid)
    lgb_train = lgb.Dataset(x_train, y_train, weight=y_train.map({0: train_w0, 1: train_w1}),
                            categorical_feature=categorical_features)
    lgb_valid = lgb.Dataset(x_valid, y_valid, weight=y_valid.map({0: valid_w0, 1: valid_w1}),
                            categorical_feature=categorical_features)

    model = lgb.train(params=params, train_set=lgb_train, num_boost_round=_num_boost_round,
                      valid_sets=[lgb_train, lgb_valid], verbose_eval=2000)
    valid_pred = model.predict(x_valid)
    return model, valid_pred


def gradient_boosting_model_cv_training(train_df: pd.DataFrame, features: list, categorical_features: list,
                                        target_col='Class', group_col='Class'):
    oof_predictions = np.zeros(len(train_df))
    models = []

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, valid_index) in enumerate(kfold.split(train_df, train_df[group_col])):
        print('-' * 50)
        print(f'Training fold: {fold + 1}')

        x_train = train_df[features].iloc[train_index]
        y_train = train_df[target_col].iloc[train_index]
        x_valid = train_df[features].iloc[valid_index]
        y_valid = train_df[target_col].iloc[valid_index]

        model, valid_pred = lightgbm_training(x_train, y_train, x_valid, y_valid, features, categorical_features)

        oof_predictions[valid_index] = valid_pred
        models.append(model)
        del x_train, x_valid, y_train, y_valid, model, valid_pred
        gc.collect()

    score = metrics.roc_auc_score(train_df[target_col], oof_predictions)
    print(f'out of folds CV is {score}')

    oof_df = pd.DataFrame({target_col: train_df[target_col], 'prediction': oof_predictions})

    return oof_df, models, score


def lightgbm_inference(x_test: pd.DataFrame, models):
    test_pred = np.zeros(len(x_test))
    for model in models:
        test_pred += model.predict(x_test)
    return test_pred / len(models)


directory = 'Data'
for dirpath, dirnames, filenames in os.walk(directory):
    for filename in filenames:
        file_path = os.path.join(dirpath, filename)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        globals()[file_name] = pd.read_csv(file_path)
        print(file_name)

train.drop(columns=['id'], inplace=True)
test.drop(columns=['id'], inplace=True)
mixed_desc.drop(columns=['CIDs'], inplace=True)

col = 'EC1_EC2_EC3_EC4_EC5_EC6'
mixed_desc[col.split('_')] = mixed_desc[col].str.split('_', expand=True).astype(int)
mixed_desc.drop(col, axis=1, inplace=True)
original = mixed_desc[train.columns]

train = pd.concat([train, original]).reset_index(drop=True)
train.drop(columns=col.split('_')[2:], inplace=True)

train_df = train.drop_duplicates()
test_df = test
submission_df = sample_submission

features = ['BertzCT', 'Chi1', 'Chi1n', 'Chi1v', 'Chi2n', 'Chi2v', 'Chi3v',
            'Chi4n', 'EState_VSA1', 'EState_VSA2', 'ExactMolWt', 'FpDensityMorgan1',
            'FpDensityMorgan2', 'FpDensityMorgan3', 'HallKierAlpha',
            'HeavyAtomMolWt', 'Kappa3', 'MaxAbsEStateIndex', 'MinEStateIndex',
            'NumHeteroatoms', 'PEOE_VSA10', 'PEOE_VSA14', 'PEOE_VSA6', 'PEOE_VSA7',
            'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3', 'VSA_EState9',
            'fr_COO', 'fr_COO2']

cat_cols = ['EState_VSA2', 'HallKierAlpha', 'NumHeteroatoms', 'PEOE_VSA10', 'PEOE_VSA14', 'PEOE_VSA6',
            'PEOE_VSA7', 'PEOE_VSA8', 'SMR_VSA10', 'SMR_VSA5', 'SlogP_VSA3', 'fr_COO', 'fr_COO2']

num_cols = [c for c in features if c not in cat_cols]
train_target = ['EC1', 'EC2']

seed_everything(42)

train_df, test_df, features_new = generate_features(train_df, test_df, cat_cols, num_cols)

train_df_num_rows = train_df.shape[0]
target_encoder = {}

for col in cat_cols:
    cnt = 0
    dic = {}
    for i, (key, value) in enumerate(collections.Counter(list(train_df[col])).most_common()):
        cnt += value
        dic[key] = i
        if cnt / train_df_num_rows > 0.7:
            break
    print(dic)

    train_df[f'{col}_cat'] = pd.Categorical(train_df[col].map(convert_to_cat))
    test_df[f'{col}_cat'] = pd.Categorical(test_df[col].map(convert_to_cat))

    target_mean = train_df.groupby(f'{col}_cat')[train_target].mean()
    target_encoder[col] = target_mean

    train_df[f'{col}_ec1_encoded'] = train_df[f'{col}_cat'].map(target_mean['EC1'])
    train_df[f'{col}_ec2_encoded'] = train_df[f'{col}_cat'].map(target_mean['EC2'])

    test_df[f'{col}_ec1_encoded'] = test_df[f'{col}_cat'].map(target_mean['EC1'])
    test_df[f'{col}_ec2_encoded'] = test_df[f'{col}_cat'].map(target_mean['EC2'])

    del (train_df[f'{col}_cat'])
    del (test_df[f'{col}_cat'])

features_new += [col for col in train_df.columns if 'encoded' in col]
print(features_new)

train_x = train_df.reset_index(drop=True)
train_y = train_df[train_target].reset_index(drop=True)

params = {'objective': 'binary', 'metric': 'auc', 'boosting': 'gbdt', 'learning_rate': 0.005, 'num_leaves': 14,
          'learning_rate': 0.010100621638956782, 'feature_fraction': 0.1492197908813077,
          'bagging_fraction': 0.27660071736347114, 'bagging_freq': 8, 'min_child_samples': 75, 'lambda_l1': 2,
          'lambda_l2': 4, 'n_jobs': -1, 'is_unbalance': True, 'verbose': -1, 'seed': 42}

_num_boost_round = 1000

train_all = train_x

oof, models_1, score = gradient_boosting_model_cv_training(train_all, features_new, [], 'EC1', 'EC1')

params = {'objective': 'binary', 'metric': 'auc', 'boosting': 'gbdt', 'learning_rate': 0.005, 'num_leaves': 5,
          'feature_fraction': 0.5, 'bagging_fraction': 0.8, 'lambda_l1': 2, 'lambda_l2': 4, 'n_jobs': -1,
          'is_unbalance': True, 'verbose': -1, 'seed': 42}

train_all = train_x

oof, models_2, score = gradient_boosting_model_cv_training(train_all, features_new, [], 'EC2', 'EC2')

test_df['EC1'] = lightgbm_inference(test_df[features_new], models_1)
test_df['EC2'] = lightgbm_inference(test_df[features_new], models_2)

tmp = pd.read_csv('Data/test.csv')

test_df['id'] = tmp['id']

test_df[list(submission_df)].to_csv('Submission.csv', index=False)

test_df[list(submission_df)].hist()
plt.show()
