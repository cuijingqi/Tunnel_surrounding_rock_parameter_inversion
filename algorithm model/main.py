import ssl
import time
import warnings
import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from skopt import BayesSearchCV

ssl._create_default_https_context = ssl._create_unverified_context
requests.packages.urllib3.disable_warnings()
warnings.filterwarnings("ignore")
from made.tunnel.tool import loaddata

scaler, head, feature_train, feature_test, target_train, target_test = loaddata.loaddata()

def mapef(y_true, y_pred):
    # print("min ape", np.min(np.abs((y_pred - y_true) / y_true)) * 100)
    # print("max ape", np.max(np.abs((y_pred - y_true) / y_true)) * 100)
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def smapef(y_true, y_pred):
    # print("min sape", np.min(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100)
    # print("max sape", np.max(2.0 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100)
    return 2.0 * np.mean(np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100


def adjusted_R2(y_true, y_pred, n, p):
    R2 = r2_score(y_true, y_pred)
    return 1 - (1 - R2) * (n - 1) / (n - p - 1)


def bsvm(C=None, gamma=None):

    if C is None and gamma is None:

        param_space = {"C": np.arange(10, 1000, 3), "gamma": np.arange(0.0005, 10, 0.008)}

        svm_model = SVR(kernel='rbf')

        bayes_cv = BayesSearchCV(svm_model, param_space, cv=10, n_iter=1000, random_state=42, n_jobs=-1)

        bayes_cv.fit(feature_train, target_train)

        print("Best parameters: ", bayes_cv.best_params_)
        print("Best score: ", bayes_cv.best_score_)
        best_param = bayes_cv.best_params_
        best_score = bayes_cv.best_score_

        best_model = bayes_cv.best_estimator_
        mse = mean_squared_error(target_test, best_model.predict(feature_test))
        r2 = r2_score(y_pred=best_model.predict(feature_test), y_true=target_test)
        rdr2 = adjusted_R2(target_test, best_model.predict(feature_test), feature_test.shape[0], feature_test.shape[1])
        mape = mapef(target_test, best_model.predict(feature_test))
        smape = smapef(target_test, best_model.predict(feature_test))
    else:

        svm_model = SVR(kernel='rbf', C=C, gamma=gamma)

        svm_model.fit(feature_train, target_train)

        mse = mean_squared_error(target_test, svm_model.predict(feature_test))
        r2 = r2_score(y_pred=svm_model.predict(feature_test), y_true=target_test)
        rdr2 = adjusted_R2(target_test, svm_model.predict(feature_test), feature_test.shape[0], feature_test.shape[1])
        mape = mapef(target_test, svm_model.predict(feature_test))
        smape = smapef(target_test, svm_model.predict(feature_test))
        best_param = {'C': C, 'gamma': gamma}
        best_score = 0
    print("SVM Mean Squared Error: ", mse)
    print("SVM R2: ", r2)
    print("SVM Adjusted R2: ", rdr2)
    print("SVM MAPE: ", mape)
    print("SVM SMAPE: ", smape)

    return smape, rdr2


def blightgbm(num_leaves=None, learning_rate=None, n_estimators=None, max_depth=None, subsample=None,
              colsample_bytree=None, min_child_samples=None):

    if num_leaves is None and learning_rate is None and n_estimators is None and max_depth is None and subsample is None and colsample_bytree is None and min_child_samples is None:

        param_space = {'num_leaves': np.arange(10, 200, 3),
                       'learning_rate': np.arange(0.005, 0.1, 0.008),
                       'n_estimators': np.arange(10, 800, 2),
                       "max_depth": np.arange(1, 10, 1),
                       'subsample': np.arange(0.5, 1, 0.1),
                       'colsample_bytree': np.arange(0.5, 1, 0.1),
                       'min_child_samples': np.arange(1, 100, 1)}

        lgb_model = LGBMRegressor(n_jobs=-1)

        bayes_cv = BayesSearchCV(lgb_model, param_space, cv=10, n_iter=1000, random_state=42, n_jobs=-1)

        bayes_cv.fit(feature_train, target_train)

        print("Best parameters: ", bayes_cv.best_params_)
        print("Best score: ", bayes_cv.best_score_)
        best_param = bayes_cv.best_params_
        best_score = bayes_cv.best_score_

        best_model = bayes_cv.best_estimator_
        mse = mean_squared_error(target_test, best_model.predict(feature_test))
        r2 = r2_score(y_pred=best_model.predict(feature_test), y_true=target_test)
        rdr2 = adjusted_R2(target_test, best_model.predict(feature_test), feature_test.shape[0], feature_test.shape[1])
        mape = mapef(target_test, best_model.predict(feature_test))
        smape = smapef(target_test, best_model.predict(feature_test))
    else:

        lgb_model = LGBMRegressor(num_leaves=num_leaves, learning_rate=learning_rate, n_estimators=n_estimators,
                                  max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree,
                                  min_child_samples=min_child_samples, n_jobs=-1)


        lgb_model.fit(feature_train, target_train)

        mse = mean_squared_error(target_test, lgb_model.predict(feature_test))
        r2 = r2_score(y_pred=lgb_model.predict(feature_test), y_true=target_test)
        rdr2 = adjusted_R2(target_test, lgb_model.predict(feature_test), feature_test.shape[0], feature_test.shape[1])
        mape = mapef(target_test, lgb_model.predict(feature_test))
        smape = smapef(target_test, lgb_model.predict(feature_test))
        best_param = {'num_leaves': num_leaves, 'learning_rate': learning_rate, 'n_estimators': n_estimators,
                      'max_depth': max_depth, 'subsample': subsample, 'colsample_bytree': colsample_bytree,
                      'min_child_samples': min_child_samples}
        best_score = 0
    print("LGB Mean Squared Error: ", mse)
    print("LGB R2: ", r2)
    print("LGB Adjusted R2: ", rdr2)
    print("LGB MAPE: ", mape)
    print("LGB SMAPE: ", smape)

    return smape, rdr2


def bgbdt(learning_rate=None, n_estimators=None, max_depth=None, subsample=None, min_samples_split=None):

    if learning_rate is None and n_estimators is None and max_depth is None and subsample is None and min_samples_split is None:

        param_space = {'learning_rate': np.arange(0.005, 0.2, 0.008),
                       'n_estimators': np.arange(10, 800, 2),
                       "max_depth": np.arange(1, 10, 1),
                       'subsample': np.arange(0.5, 1, 0.1),
                       'min_samples_split': np.arange(2, 20)}

        gbdt_model = GBDT = GradientBoostingRegressor()

        bayes_cv = BayesSearchCV(gbdt_model, param_space, cv=10, n_iter=1000, random_state=42, n_jobs=-1)

        bayes_cv.fit(feature_train, target_train)

        print("Best parameters: ", bayes_cv.best_params_)
        print("Best score: ", bayes_cv.best_score_)
        best_param = bayes_cv.best_params_
        best_score = bayes_cv.best_score_

        best_model = bayes_cv.best_estimator_
        mse = mean_squared_error(target_test, best_model.predict(feature_test))
        r2 = r2_score(y_pred=best_model.predict(feature_test), y_true=target_test)
        rdr2 = adjusted_R2(target_test, best_model.predict(feature_test), feature_test.shape[0], feature_test.shape[1])
        mape = mapef(target_test, best_model.predict(feature_test))
        smape = smapef(target_test, best_model.predict(feature_test))
    else:

        gbdt_model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators,
                                               max_depth=max_depth, subsample=subsample,
                                               min_samples_split=min_samples_split)

        gbdt_model.fit(feature_train, target_train)

        mse = mean_squared_error(target_test, gbdt_model.predict(feature_test))
        r2 = r2_score(y_pred=gbdt_model.predict(feature_test), y_true=target_test)
        rdr2 = adjusted_R2(target_test, gbdt_model.predict(feature_test), feature_test.shape[0], feature_test.shape[1])
        mape = mapef(target_test, gbdt_model.predict(feature_test))
        smape = smapef(target_test, gbdt_model.predict(feature_test))
        best_param = {'learning_rate': learning_rate, 'n_estimators': n_estimators,
                      'max_depth': max_depth, 'subsample': subsample,
                      'min_samples_split': min_samples_split}
        best_score = 0
    print("GBDT Mean Squared Error: ", mse)
    print("GBDT R2: ", r2)
    print("GBDT Adjusted R2: ", rdr2)
    print("GBDT MAPE: ", mape)
    print("GBDT SMAPE: ", smape)

    return smape, rdr2


def bxgboost(learning_rate=None, n_estimators=None, max_depth=None, subsample=None, colsample_bytree=None):

    if learning_rate is None and n_estimators is None and max_depth is None and subsample is None and colsample_bytree is None:

        param_space = {'learning_rate': np.arange(0.005, 0.2, 0.008),
                       'n_estimators': np.arange(10, 800, 2),
                       "max_depth": np.arange(1, 10, 1),
                       'subsample': np.arange(0.5, 1, 0.1),
                       'colsample_bytree': np.arange(0.5, 1, 0.1)}

        xgb_model = xgb.XGBRegressor(n_jobs=-1)

        bayes_cv = BayesSearchCV(xgb_model, param_space, cv=10, n_iter=1000, random_state=42, n_jobs=-1)

        bayes_cv.fit(feature_train, target_train)

        print("Best parameters: ", bayes_cv.best_params_)
        print("Best score: ", bayes_cv.best_score_)
        best_param = bayes_cv.best_params_
        best_score = bayes_cv.best_score_

        best_model = bayes_cv.best_estimator_
        mse = mean_squared_error(target_test, best_model.predict(feature_test))
        r2 = r2_score(y_pred=best_model.predict(feature_test), y_true=target_test)
        rdr2 = adjusted_R2(target_test, best_model.predict(feature_test), feature_test.shape[0], feature_test.shape[1])
        mape = mapef(target_test, best_model.predict(feature_test))
        smape = smapef(target_test, best_model.predict(feature_test))
    else:

        xgb_model = xgb.XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators,
                                     max_depth=max_depth, subsample=subsample,
                                     colsample_bytree=colsample_bytree)

        xgb_model.fit(feature_train, target_train)

        mse = mean_squared_error(target_test, xgb_model.predict(feature_test))
        r2 = r2_score(y_pred=xgb_model.predict(feature_test), y_true=target_test)
        rdr2 = adjusted_R2(target_test, xgb_model.predict(feature_test), feature_test.shape[0], feature_test.shape[1])
        mape = mapef(target_test, xgb_model.predict(feature_test))
        smape = smapef(target_test, xgb_model.predict(feature_test))
        best_param = {'learning_rate': learning_rate, 'n_estimators': n_estimators,
                      'max_depth': max_depth, 'subsample': subsample,
                      'colsample_bytree': colsample_bytree}
        best_score = 0
    print("XGB Mean Squared Error: ", mse)
    print("XGB R2: ", r2)
    print("XGB Adjusted R2: ", rdr2)
    print("XGB MAPE: ", mape)
    print("XGB SMAPE: ", smape)

    return smape, rdr2


def brf(n_estimators=None, max_depth=None, min_samples_leaf=None, min_samples_split=None, max_features=None):

    if n_estimators is None and max_depth is None and min_samples_leaf is None and min_samples_split is None and max_features is None:

        param_space = {"n_estimators": np.arange(10, 800, 2),
                       'max_depth': np.arange(1, 20, 1),
                       "min_samples_leaf": np.arange(1, 20),
                       "min_samples_split": np.arange(2, 20),
                       'max_features': np.arange(1, 3)}

        rf_model = RandomForestRegressor(n_jobs=-1)

        bayes_cv = BayesSearchCV(rf_model, param_space, cv=10, n_iter=1000, random_state=42, n_jobs=-1)

        bayes_cv.fit(feature_train, target_train)

        print("Best parameters: ", bayes_cv.best_params_)
        print("Best score: ", bayes_cv.best_score_)
        best_param = bayes_cv.best_params_
        best_score = bayes_cv.best_score_

        best_model = bayes_cv.best_estimator_
        mse = mean_squared_error(target_test, best_model.predict(feature_test))
        r2 = r2_score(y_pred=best_model.predict(feature_test), y_true=target_test)
        rdr2 = adjusted_R2(target_test, best_model.predict(feature_test), feature_test.shape[0], feature_test.shape[1])
        mape = mapef(target_test, best_model.predict(feature_test))
        smape = smapef(target_test, best_model.predict(feature_test))
    else:

        rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                         min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                         max_features=max_features)

        rf_model.fit(feature_train, target_train)

        mse = mean_squared_error(target_test, rf_model.predict(feature_test))
        r2 = r2_score(y_pred=rf_model.predict(feature_test), y_true=target_test)
        rdr2 = adjusted_R2(target_test, rf_model.predict(feature_test), feature_test.shape[0], feature_test.shape[1])
        mape = mapef(target_test, rf_model.predict(feature_test))
        smape = smapef(target_test, rf_model.predict(feature_test))
        best_param = {'n_estimators': n_estimators, 'max_depth': max_depth,
                      'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
                      'max_features': max_features}
        best_score = 0
    print("RF Mean Squared Error: ", mse)
    print("RF R2: ", r2)
    print("RF Adjusted R2: ", rdr2)
    print("RF MAPE: ", mape)
    print("RF SMAPE: ", smape)

    return smape, rdr2


clfa = []


def explain(svr_model, lgb_model, gbdt_model, xgb_model, rf_model):
    import warnings

    import lime
    import numpy as np

    from sklearn.ensemble import VotingRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    from made.tunnel.tool import loaddata


    warnings.filterwarnings("ignore", message="Found `num_iterations` in params. Will use it instead of argument")

    scaler, head, feature_train, feature_test, target_train, target_test = loaddata.loaddata()
    
    estimators = [('svr_model', svr_model), ('lgb_model', lgb_model), ("gbdt_model", gbdt_model),
                  ("xgb_model", xgb_model),
                  ("rf_model", rf_model)]
   
    for clf in estimators:
        clf[1].fit(feature_train, target_train)
 
    explainer = lime.lime_tabular.LimeTabularExplainer(feature_train, mode="regression", feature_names=head[-3:],
                                                       class_names=['Deformation modulus'],
                                                       verbose=False)

    t = 1000 * time.time()
    np.random.seed(int(t) % 2 ** 32)

    sample_feature_train = feature_train[
        np.random.choice(feature_train.shape[0], int(feature_train.shape[0] * 0.2), replace=False)]

    clf_allimportances = []
    for clf in estimators:
        clf_importances = []

        for i in range(sample_feature_train.shape[0]):
            exp = explainer.explain_instance(sample_feature_train[i], clf[1].predict, num_features=3)
            clf_importance = [exp.as_list()[j][1] for j in range(len(exp.as_list()))]
            clf_importances.append(clf_importance)

        clf_importances = np.array(clf_importances)
        clf_importances = clf_importances.mean(axis=0)
        clf_allimportances.append(clf_importances)

    clf_allimportances = np.array(clf_allimportances)
    clf_allimportances = clf_allimportances.sum(axis=1)
    clf_allimportances = np.abs(clf_allimportances)

    clfa.append(clf_allimportances)

    clf_weights = clf_allimportances / clf_allimportances.sum(axis=0)

    voting_clf = VotingRegressor(estimators=estimators, weights=clf_weights)
    voting_clf.fit(feature_train, target_train)

    PredictTrain = voting_clf.predict(feature_train)
    PredictTest = voting_clf.predict(feature_test)
    mse = mean_squared_error(target_test, PredictTest)
    r2 = r2_score(y_pred=PredictTest, y_true=target_test)
    rdr2 = adjusted_R2(target_test, PredictTest, feature_test.shape[0], feature_test.shape[1])
    mape = mapef(target_test, PredictTest)
    smape = smapef(target_test, PredictTest)
    # print("Explain Mean Squared Error: ", mse)
    # print("Explain R2: ", r2)
    # print("Explain Adjusted R2: ", rdr2)
    # print("Explain MAPE: ", mape)
    # print("Explain SMAPE: ", smape)
    return smape, rdr2


if __name__ == '__main__':
