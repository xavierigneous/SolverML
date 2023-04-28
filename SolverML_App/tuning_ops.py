from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, LeaveOneOut, ShuffleSplit, RepeatedKFold, cross_val_predict, cross_validate
# import dask_ml
# import dask_ml.model_selection as dcv
from distributed import Client
import joblib, time
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier, SGDClassifier, SGDRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestClassifier, \
    AdaBoostClassifier, AdaBoostRegressor, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import numpy as np

def classify_metric(Y):
    # Used to detect binary/non-binary classification to choose appropriate scoring
    if len(set(Y)) <= 2:
        return 'roc_auc'
    else:
        return 'f1_weighted'

# -------------Regression-------------------------------------------------------
#Number of folds
class_split = {'kfold': KFold(n_splits=3, shuffle=True),
             'stratifiedk_fold': StratifiedKFold(n_splits=3, shuffle=True)}
reg_split = {'kfold': KFold(n_splits=3, shuffle=True),
     'shuffle_split': ShuffleSplit(n_splits=3)}
def get_split(problem_type,folds):
    class_split = {'kfold': KFold(n_splits=folds, shuffle=True),
             'stratifiedk_fold': StratifiedKFold(n_splits=folds, shuffle=True)}
    reg_split = {'kfold': KFold(n_splits=folds, shuffle=True),
         'shuffle_split': ShuffleSplit(n_splits=folds)}
    split_type = class_split if problem_type=='classification' else reg_split
    return split_type

global client
# client = Client(), scheduler=client


def rf_regress(X, y, search_type, validation_type, problem_type, folds):
    rf_params = {
        'n_estimators': [10, 20, 30],
        'max_features': ['sqrt', 0.5],
        'max_depth': [15, 20, 30, 50],
        'min_samples_leaf': [1, 2, 4, 8],
        "criterion": ['mse', 'mae']
    }

    split_type = get_split(problem_type,folds)
    cv = split_type.get(validation_type)
    clf = RandomForestRegressor(random_state=0)
    # client = Client()
    search = {
        'GridSearchCV': GridSearchCV(clf, rf_params, cv=cv, scoring='neg_mean_absolute_error'),
        'RandomisedSearchCV': RandomizedSearchCV(clf, rf_params, cv,
                                                     scoring='neg_mean_absolute_error')}

    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid = search.get(search_type)
    grid.fit(X, y)
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = RandomForestRegressor(**grid.best_params_)
    model.fit(X, y)
    model_name='Tuned_RandomForestRegressor '+search_type+' '+validation_type+' '+str(folds)+' folds'
    return grid.best_params_, grid.best_score_, model_name, model


def xgb_regress(X, y, search_type, validation_type, problem_type, folds):
    xgb_params = {'max_depth': list(range(3, 10, 2)),
                  'min_child_weight': [1, 5, 20, 50],
                  'gamma': [i / 10. for i in range(0, 4)],
                  'subsample': [i / 10.0 for i in range(6, 10)],
                  'colsample_bytree': [i / 10.0 for i in range(6, 10)],
                  'reg_alpha': [1e-6, 1e-2, 0.1, 1, 100],
                  'reg_lambda': [1e-6, 1e-2, 0.1, 1, 100]}
    split_type = class_split if problem_type=='classification' else reg_split
    cv = split_type.get(validation_type)
    clf = XGBRegressor(random_state=0)
    # client = Client()
    search = {
        'GridSearchCV': GridSearchCV(clf, xgb_params, cv=cv, scoring='neg_mean_absolute_error', verbose=2),
        'RandomisedSearchCV': RandomizedSearchCV(clf, xgb_params, cv=cv, scoring='neg_mean_absolute_error')}
    grid = search.get(search_type)
    clf.fit(X, y)
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(classify_metric(y))
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = XGBRegressor(**grid.best_params_)
    model.fit(X, y)
    #joblib.dump(model, os.path.join(temp_dir, 'Tuned_XGBRegressor.sav'))
    model_name='Tuned_KNeighborsRegressor '+search_type+' '+validation_type+' '+str(folds)+' folds'
    return grid.best_params_, grid.best_score_, model_name, model


def log_regress(X, y, search_type, validation_type, problem_type, folds):
    log_reg_params = {'penalty': 'l2',
                      'C': [0.001, 0.01, 0.1, 1, 10, 100],
                      'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                      'penalty': ['none', 'l1', 'l2', 'elasticnet']
                      }
    split_type = class_split if problem_type=='classification' else reg_split
    cv = split_type.get(validation_type)
    clf = LinearRegression()
    # client = Client()
    search = {'GridSearchCV': GridSearchCV(clf, log_reg_params, cv=cv, scoring='neg_root_mean_squared_error'),
              'RandomisedSearchCV': RandomizedSearchCV(clf, log_reg_params, cv=cv,
                                                           scoring='neg_root_mean_squared_error')}
    grid = search.get(search_type)
    # grid = GridSearchCV(clf, log_reg_params, cv=3, scoring='neg_root_mean_squared_error')
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = LinearRegression(**grid.best_params_)
    model.fit(X, y)
    #joblib.dump(model, os.path.join(temp_dir, 'Tuned_LinearRegression.sav'))
    model_name='Tuned_LinearRegression '+search_type+' '+validation_type+' '+str(folds)+' folds'
    return grid.best_params_, grid.best_score_, model_name, model


def knn_regress(X, y, search_type, validation_type, problem_type, folds):
    knn_params = {
        'n_neighbors': [2, 3, 5, 10, 15, 20],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'weights': ['uniform', 'distance']
    }
    split_type = class_split if problem_type=='classification' else reg_split
    cv = split_type.get(validation_type)
    clf = KNeighborsRegressor()
    # client = Client()
    search = {'GridSearchCV': GridSearchCV(clf, knn_params, cv=cv, scoring='neg_root_mean_squared_error'),
              'RandomisedSearchCV': RandomizedSearchCV(clf, knn_params, cv=cv,
                                                           scoring='neg_root_mean_squared_error',)}
    grid = search.get(search_type)
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = KNeighborsRegressor(**grid.best_params_)
    model.fit(X, y)
    #joblib.dump(model, os.path.join(temp_dir, 'Tuned_KNeighborsRegressor.sav'))
    model_name='Tuned_KNeighborsRegressor '+search_type+' '+validation_type+' '+str(folds)+' folds'
    return grid.best_params_, grid.best_score_, model_name, model


def svc_regress(X, y, search_type, validation_type, problem_type, folds):
    svc_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1]
                  # "kernel":['linear','rbf']
                  }
    split_type = class_split if problem_type=='classification' else reg_split
    cv = split_type.get(validation_type)
    clf = SVR(gamma='scale')
    # client = Client()
    search = {'GridSearchCV': GridSearchCV(clf, svc_params, cv=cv, scoring='neg_root_mean_squared_error'),
              'RandomisedSearchCV': RandomizedSearchCV(clf, svc_params, cv=cv,
                                                           scoring='neg_root_mean_squared_error',)}
    grid = search.get(search_type)
    # grid = GridSearchCV(clf, svc_params, cv=3, scoring='neg_root_mean_squared_error')
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = SVR(**grid.best_params_)
    model.fit(X, y)
    #joblib.dump(model, os.path.join(temp_dir, 'Tuned_SVR.sav'))
    model_name='Tuned_SVR '+search_type+' '+validation_type+' '+str(folds)+' folds'
    return grid.best_params_, grid.best_score_, model_name, model

def dtree_regress(X, y, search_type, validation_type, problem_type, folds):
    # create a dictionary of all values we want to test
    dtree_param = {'criterion': ['mse', 'mae'], 'max_depth': np.arange(3, 15)}
    split_type = get_split(problem_type,folds)
    cv = split_type.get(validation_type)
    dtree_model = DecisionTreeRegressor()
    ## client = Client()
    search = {'GridSearchCV': GridSearchCV(dtree_model, dtree_param, cv=cv, scoring='neg_root_mean_squared_error'),
              'RandomisedSearchCV': RandomizedSearchCV(dtree_model, dtree_param, cv=cv,
                                                           scoring='neg_root_mean_squared_error')}
    grid = search.get(search_type)
    print('Fold : ',cv)
    # grid = GridSearchCV(dtree_model, dtree_param, cv=3, scoring='neg_root_mean_squared_error')
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = DecisionTreeRegressor(**grid.best_params_)
    model.fit(X, y)
    model_name='Tuned DecisionTreeRegressor '+search_type+' '+validation_type+' '+str(folds)+' folds'
    #joblib.dump(model, os.path.join(temp_dir, ))
    return grid.best_params_, grid.best_score_, model_name, model


def adaboost_regress(X, y, search_type, validation_type, problem_type, folds):
    # create a dictionary of all values we want to test
    dtree_param = {'n_estimators': [50, 100, 200, 500, 1000],
                   'learning_rate': [0.01, 0.1, 1., 2.]}
    split_type = class_split if problem_type=='classification' else reg_split
    cv = split_type.get(validation_type)
    dtree_model = AdaBoostRegressor()
    
    search = {'GridSearchCV': GridSearchCV(dtree_model, dtree_param, cv=cv, scoring='neg_root_mean_squared_error'),
              'RandomisedSearchCV': RandomizedSearchCV(dtree_model, dtree_param, cv=cv,
                                                           scoring='neg_root_mean_squared_error')}
    grid = search.get(search_type)
    # grid = GridSearchCV(dtree_model, dtree_param, cv=3, scoring='neg_root_mean_squared_error')
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = AdaBoostRegressor(**grid.best_params_)
    model.fit(X, y)
    #joblib.dump(model, os.path.join(temp_dir, 'Tuned_AdaBoostRegressor.sav'))
    model_name='Tuned_AdaBoostRegressor '+search_type+' '+validation_type+' '+str(folds)+' folds'
    return grid.best_params_, grid.best_score_, model_name, model


def gradboost_regress(X, y, search_type, validation_type, problem_type, folds):
    # create a dictionary of all values we want to test
    dtree_param = {'n_estimators': [50, 100, 200, 500, 1000],
                   'learning_rate': [0.01, 0.1, 1., 2.],
                   'max_depth': [4, 6, 8]}
    split_type = class_split if problem_type=='classification' else reg_split
    cv = split_type.get(validation_type)
    dtree_model = GradientBoostingRegressor()
    # client = Client()
    search = {'GridSearchCV': GridSearchCV(dtree_model, dtree_param, cv=cv, scoring='neg_root_mean_squared_error'),
              'RandomisedSearchCV': RandomizedSearchCV(dtree_model, dtree_param, cv=cv,
                                                           scoring='neg_root_mean_squared_error')}
    grid = search.get(search_type)
    # grid = GridSearchCV(dtree_model, dtree_param, cv=3, scoring='neg_root_mean_squared_error')
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = GradientBoostingRegressor(**grid.best_params_)
    model.fit(X, y)
    #joblib.dump(model, os.path.join(temp_dir, 'Tuned_AdaBoostRegressor.sav'))
    model_name='Tuned_GradientBoostRegressor '+search_type+' '+validation_type+' '+str(folds)+' folds'
    return grid.best_params_, grid.best_score_, model_name, model


# -------------Classification-----------------------------------------
def rf_classify(X, y, search_type, validation_type, problem_type, folds):
    rf_params = {
        'n_estimators': [10, 20, 30],
        'max_features': ['sqrt', 0.5],
        'max_depth': [15, 20, 30, 50],
        'min_samples_leaf': [1, 2, 4, 8],
        # "bootstrap":[True,False],
        "criterion": ['gini', 'entropy']
    }
    split_type = class_split if problem_type=='classification' else reg_split
    cv = split_type.get(validation_type)
    dtree_model = RandomForestClassifier(random_state=0)
    # client = Client()
    search = {
        'GridSearchCV': GridSearchCV(dtree_model, rf_params, cv=cv, scoring=classify_metric(y)),
        'RandomisedSearchCV': RandomizedSearchCV(dtree_model, rf_params, cv=cv, scoring=classify_metric(y))}
    grid = search.get(search_type)
    # grid = GridSearchCV(clf, rf_params, cv=3, scoring=classify_metric(y))
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(classify_metric(y))
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = RandomForestClassifier(**grid.best_params_)
    model.fit(X, y)
    #joblib.dump(model, os.path.join(temp_dir, 'Tuned_RandomForestClassifier.sav'))
    model_name='Tuned_RandomForestClassifier'+search_type+' '+validation_type+' '+str(folds)+' folds'
    print(model_name)
    return grid.best_params_, grid.best_score_, model_name, model


def xgb_classify(X, y, search_type, validation_type, problem_type, folds):
    xgb_params = {'max_depth': list(range(3, 10, 2)),
                  'min_child_weight': [1, 5, 20, 50],
                  'gamma': [i / 10. for i in range(0, 4)],
                  'subsample': [i / 10.0 for i in range(6, 10)],
                  'colsample_bytree': [i / 10.0 for i in range(6, 10)],
                  'reg_alpha': [1e-6, 1e-2, 0.1, 1, 100],
                  'reg_lambda': [1e-6, 1e-2, 0.1, 1, 100]}
    split_type = class_split if problem_type=='classification' else reg_split
    cv = split_type.get(validation_type)
    dtree_model = XGBClassifier(random_state=0)
    # client = Client()
    search = {
        'GridSearchCV': GridSearchCV(dtree_model, xgb_params, cv=cv, scoring=classify_metric(y)),
        'RandomisedSearchCV': RandomizedSearchCV(dtree_model, xgb_params, cv=cv, scoring=classify_metric(y))}
    grid = search.get(search_type)
    # grid = GridSearchCV(clf, xgb_params, cv=3, scoring=classify_metric(y))
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(classify_metric(y))
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = XGBClassifier(**grid.best_params_)
    model.fit(X, y)
    #joblib.dump(model, os.path.join(temp_dir, 'Tuned_XGBClassifier.sav'))
    model_name='Tuned_XGBClassifier '+search_type+' '+validation_type+' '+str(folds)+' folds'
    return grid.best_params_, grid.best_score_, model_name, model


def log_classify(X, y, search_type, validation_type, problem_type, folds):
    log_reg_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                      #'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                      'penalty': ['none', 'l1', 'l2', 'elasticnet']
                      }
    split_type = class_split if problem_type=='classification' else reg_split
    cv = split_type.get(validation_type)
    dtree_model = LogisticRegression()
    # client = Client()
    search = {'GridSearchCV': GridSearchCV(dtree_model, log_reg_params, cv=cv, scoring=classify_metric(y)),
              'RandomisedSearchCV': RandomizedSearchCV(dtree_model, log_reg_params, cv=cv,
                                                           scoring=classify_metric(y))}
    grid = search.get(search_type)
    # grid = GridSearchCV(clf, log_reg_params, cv=3, scoring=classify_metric(y))
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(classify_metric(y))
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = LogisticRegression(**grid.best_params_)
    model.fit(X, y)
    #joblib.dump(model, os.path.join(temp_dir, 'Tuned_LogisticRegression.sav'))

    model_name='Tuned_LogisticRegression'+search_type+' '+validation_type+' '+str(folds)+' folds'
    return grid.best_params_, grid.best_score_, model_name, model


def knn_classify(X, y, search_type, validation_type, problem_type, folds):
    knn_params = {
        'n_neighbors': [2, 3, 5, 10, 15, 20],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'weights': ['uniform', 'distance']
    }
    split_type = class_split if problem_type=='classification' else reg_split
    cv = split_type.get(validation_type)
    dtree_model = KNeighborsClassifier()
    # client = Client()
    search = {
        'GridSearchCV': GridSearchCV(dtree_model, knn_params, cv=cv, scoring=classify_metric(y)),
        'RandomisedSearchCV': RandomizedSearchCV(dtree_model, knn_params, cv=cv, scoring=classify_metric(y))}
    grid = search.get(search_type)
    # grid = GridSearchCV(clf, knn_params, cv=3, scoring=classify_metric(y))
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(classify_metric(y))
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = KNeighborsClassifier(**grid.best_params_)
    model.fit(X, y)
    #joblib.dump(model, os.path.join(temp_dir, 'Tuned_KNeighborsClassifier.sav'))
    model_name='Tuned_KNeighborsClassifier '+search_type+' '+validation_type+' '+str(folds)+' folds'
    return grid.best_params_, grid.best_score_, model_name, model


def svc_classify(X, y, search_type, validation_type, problem_type, folds):
    svc_params = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'gamma': [0.001, 0.01, 0.1, 1]
                  # "kernel":['linear','rbf']
                  }
    split_type = class_split if problem_type=='classification' else reg_split
    cv = split_type.get(validation_type)
    dtree_model = SVC(gamma='scale')
    # client = Client()
    search = {
        'GridSearchCV': GridSearchCV(dtree_model, svc_params, cv=cv, scoring=classify_metric(y)),
        'RandomisedSearchCV': RandomizedSearchCV(dtree_model, svc_params, cv=cv, scoring=classify_metric(y))}
    grid = search.get(search_type)
    # grid = GridSearchCV(clf, svc_params, cv=3, scoring=classify_metric(y))
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(classify_metric(y))
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = SVC(**grid.best_params_)
    model.fit(X, y)
    #joblib.dump(model, os.path.join(temp_dir, 'Tuned_SVC.sav'))
    model_name='Tuned_SVC '+search_type+' '+validation_type+' '+str(folds)+' folds'
    return grid.best_params_, grid.best_score_, model_name, model


def dtree_classify(X, y, search_type, validation_type, problem_type, folds):
    # create a dictionary of all values we want to test
    dtree_param = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(3, 15)}
    split_type = class_split if problem_type=='classification' else reg_split
    cv = split_type.get(validation_type)
    dtree_model = DecisionTreeClassifier()
    # client = Client()
    search = {
        'GridSearchCV': GridSearchCV(dtree_model, dtree_param, cv=cv, scoring=classify_metric(y)),
        'RandomisedSearchCV': RandomizedSearchCV(dtree_model, dtree_param, cv=cv, scoring=classify_metric(y))}
    grid = search.get(search_type)
    # grid = GridSearchCV(dtree_model, dtree_param, cv=3, scoring=classify_metric(y))
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(classify_metric(y))
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = DecisionTreeClassifier(**grid.best_params_)
    model.fit(X, y)
    #joblib.dump(model, os.path.join(temp_dir, 'Tuned_DecisionTreeClassifier.sav'))
    model_name='Tuned_DecisionTreeClassifier '+search_type+' '+validation_type+' '+str(folds)+' folds'
    return grid.best_params_, grid.best_score_, model_name, model


def adaboost_classify(X, y, search_type, validation_type, problem_type, folds):
    # create a dictionary of all values we want to test
    dtree_param = {'n_estimators': [50, 100, 200, 500, 1000],
                   'learning_rate': [0.01, 0.1, 1., 2.]}
    split_type = class_split if problem_type=='classification' else reg_split
    cv = split_type.get(validation_type)
    dtree_model = AdaBoostClassifier()
    # client = Client()
    search = {
        'GridSearchCV': GridSearchCV(dtree_model, dtree_param, cv=cv, scoring=classify_metric(y)),
        'RandomisedSearchCV': RandomizedSearchCV(dtree_model, dtree_param, cv=cv, scoring=classify_metric(y))}
    grid = search.get(search_type)
    # grid = GridSearchCV(dtree_model, dtree_param, cv=3, scoring=classify_metric(y))
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(classify_metric(y))
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = AdaBoostClassifier(**grid.best_params_)
    model.fit(X, y)
    #joblib.dump(model, os.path.join(temp_dir, 'Tuned_AdaBoostClassifier.sav'))
    model_name='Tuned_AdaBoostClassifier '+search_type+' '+validation_type+' '+str(folds)+' folds'
    return grid.best_params_, grid.best_score_, model_name, model


def gradboost_classify(X, y, search_type, validation_type, problem_type, folds):
    # create a dictionary of all values we want to test
    dtree_param = {'n_estimators': [50, 100, 200, 500, 1000],
                   'learning_rate': [0.01, 0.1, 1., 2.],
                   'max_depth': [4, 6, 8]}
    split_type = class_split if problem_type=='classification' else reg_split
    cv = split_type.get(validation_type)
    dtree_model = GradientBoostingClassifier()
    # client = Client()
    search = {
        'GridSearchCV': GridSearchCV(dtree_model, dtree_param, cv=cv, scoring=classify_metric(y)),
        'RandomisedSearchCV': RandomizedSearchCV(dtree_model, dtree_param, cv=cv, scoring=classify_metric(y))}
    grid = search.get(search_type)
    # grid = GridSearchCV(dtree_model, dtree_param, cv=3, scoring=classify_metric(y))
    start_time = time.time()
    # with joblib.parallel_backend('dask'):
    grid.fit(X, y)
    print(classify_metric(y))
    print(grid.best_params_)
    print(grid.best_score_)
    print("--- Fit Time: %s seconds ---" % (time.time() - start_time))
    model = GradientBoostingClassifier(**grid.best_params_)
    model.fit(X, y)
    #joblib.dump(model, os.path.join(temp_dir, 'Tuned_AdaBoostClassifier.sav'))
    model_name='Tuned_GradientBoostClassifier '+search_type+' '+validation_type+' '+str(folds)+' folds'
    return grid.best_params_, grid.best_score_, model_name, model