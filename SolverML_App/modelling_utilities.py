import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, \
    KFold, StratifiedKFold, LeaveOneOut, ShuffleSplit, RepeatedKFold, \
        cross_val_predict, cross_validate, train_test_split

#Classification Metrics
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

#Regression Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

#Imbalance
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE

classification_scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted', 'roc_auc_ovr_weighted']
regression_scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2','explained_variance']

def fit_ml_algo(algo, x, y, fold, problem_type, fold_type):
    print(fold_type)
    split = {'kfold': KFold(n_splits=fold, shuffle=True),
             'stratifiedk_fold': StratifiedKFold(n_splits=fold, shuffle=True),
             'shuffle_split': ShuffleSplit(n_splits=fold)}
    cv = split.get(fold_type)
    scoring = regression_scoring if problem_type == 'regression' else classification_scoring
    scores =   cross_val_predict(algo, x, y, cv=cv, scoring=scoring, return_train_score=False)
    return scores


def fit_ml_algo_predict(algo, x, y, fold, problem_type, fold_type):
    print(fold_type)
    split = {'kfold': KFold(n_splits=fold, shuffle=True),
             'stratifiedk_fold': StratifiedKFold(n_splits=fold, shuffle=True),
             'shuffle_split': ShuffleSplit(n_splits=fold)}
    cv = split.get(fold_type)
    print(cv)
    scoring = regression_scoring if problem_type == 'regression' else classification_scoring
    y_pred =   cross_validate(algo, x, y, cv=cv, scoring=scoring, return_train_score=False)
    return y_pred


def roc_auc_score_multiclass(test, pred, average):
    uniques = set(test)
    roc_auc_dict = {}
    for class_obs in uniques:
        others = [x for x in uniques if x != class_obs]
        new_test = [0 if x in others else 1 for x in test]
        new_pred = [0 if x in others else 1 for x in pred]
        roc_auc =  roc_auc_score(new_test, new_pred, average=average)
        roc_auc_dict[class_obs] = roc_auc

    return np.mean(list(roc_auc_dict.values()))


def train_test_fit(model, x, y, test_size, problem_type, smote):
    smote = smote if problem_type == 'classification' else ''
    stratify = y if problem_type == 'classification' else None
    X_train, X_test, Y_train, Y_test =   train_test_split(x, y, test_size=test_size / 100,
                                                                        random_state=100, stratify=stratify)

    smote_grid = {'SMOTE': SMOTE(random_state=10), 'SMOTEENN': SMOTEENN(random_state=10)}
    if smote != '':
        print('SMOTE: ', smote_grid.get(smote))
        sm = smote_grid.get(smote)
        X_train, Y_train = sm.fit_resample(X_train, Y_train)
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    classification_scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted',
                              'roc_auc_ovr_weighted']
    regression_scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2', 'explained_variance']
    if problem_type == 'classification':
        acc = [ accuracy_score(Y_test, Y_pred)]
        print('Number of Target Labels: ', len(set(Y_train)))
        if len(set(Y_train)) <= 2:
            roc_auc = [ roc_auc_score(Y_test, Y_pred, average='weighted')]
        else:
            roc_auc = [roc_auc_score_multiclass(Y_test, Y_pred, average='weighted')]

        precision = [ precision_score(Y_test, Y_pred, average='weighted', labels=np.unique(Y_pred))]
        recall = [ recall_score(Y_test, Y_pred, average='weighted', labels=np.unique(Y_pred))]
        f1 = [ f1_score(Y_test, Y_pred, average='weighted', labels=np.unique(Y_pred))]
        scoring = classification_scoring
        class_result = pd.DataFrame(list(zip(acc, roc_auc, precision, recall, f1)),
                                    columns=scoring)
    if problem_type == 'regression':
        neg_mean_absolute_error = [ mean_absolute_error(Y_test, Y_pred)]
        neg_mean_squared_error = [ mean_squared_error(Y_test, Y_pred)]
        neg_root_mean_squared_error = [ mean_squared_error(Y_test, Y_pred, squared=False)]
        r2 = [ r2_score(Y_test, Y_pred)]
        explained_variance = [ explained_variance_score(Y_test, Y_pred)]
        scoring = regression_scoring
        reg_result = pd.DataFrame(
            list(zip(neg_mean_absolute_error, neg_mean_squared_error, neg_root_mean_squared_error, r2, explained_variance)),
            columns=scoring)

    res = reg_result if problem_type == 'regression' else class_result
    print(res)
    return res