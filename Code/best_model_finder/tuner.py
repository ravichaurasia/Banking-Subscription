from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
# import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from xgboost import XGBClassifier
import lightgbm as lgboost
import catboost as cboost
from Training_Testing_Performance import PerformanceEvaluation

class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Version: 1.0
                Revisions: None
    """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.clf = RandomForestClassifier()
        self.DecisionTreeReg = DecisionTreeRegressor()
        self.score = PerformanceEvaluation.performance(self.file_object, self.logger_object)

    def get_best_params_for_random_forest(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5,  verbose=3)

            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_DecisionTreeRegressor(self, train_x, train_y):
        """
                                                Method Name: get_best_params_for_DecisionTreeRegressor
                                                Description: get the parameters for DecisionTreeRegressor Algorithm which give the best accuracy.
                                                             Use Hyper Parameter Tuning.
                                                Output: The model with the best parameters
                                                On Failure: Raise Exception

                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_DecisionTreeRegressor method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_decisionTree = {"criterion": ["mse", "friedman_mse", "mae"],
                              "splitter": ["best", "random"],
                              "max_features": ["auto", "sqrt", "log2"],
                              'max_depth': range(2, 16, 2),
                              'min_samples_split': range(2, 16, 2)
                              }

            # Creating an object of the Grid Search class
            self.grid = GridSearchCV(self.DecisionTreeReg, self.param_grid_decisionTree, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.splitter = self.grid.best_params_['splitter']
            self.max_features = self.grid.best_params_['max_features']
            self.max_depth  = self.grid.best_params_['max_depth']
            self.min_samples_split = self.grid.best_params_['min_samples_split']

            # creating a new model with the best parameters
            self.decisionTreeReg = DecisionTreeRegressor(criterion=self.criterion,splitter=self.splitter,max_features=self.max_features,max_depth=self.max_depth,min_samples_split=self.min_samples_split)
            # training the mew models
            self.decisionTreeReg.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Decision-Tree Regressor best params: ' + str(
                                       self.grid.best_params_) + '. Exited method of the Model fit')
            return self.decisionTreeReg
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in Decision-Tree Regressor method of the Model Fit. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Grid search Parameter tuning  failed. Exited the Decision-Tree Regressor method of the Model Fit')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBRegressor(objective='reg:linear'),self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBRegressor(objective='reg:linear',learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()

    def xgb_classifier(self, train_x, train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.

                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # creating a new model with the best parameters
            self.xgbc = XGBClassifier(objective='binary:logistic')
                        # training the mew model
            self.xgbc.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost model train method done of the Model_Finder class')
            return self.xgbc
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()
    def lgb_classifier(self, train_x, train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.

                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the lgb_classifier class')
        try:
            # creating a new model with the best parameters
            self.lgbc = lgboost.LGBMClassifier()
                        # training the mew model
            self.lgbc.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'LGBoost model train method done of the Model_Finder class')
            return self.lgbc
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_LGBboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'LGBoost Parameter tuning  failed. Exited the get_best_params_for_LGboost method of the Model_Finder class')
            raise Exception()

    def catb_classifier(self, train_x, train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.

                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Version: 1.0
                                        Revisions: None
                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # creating a new model with the best parameters

            self.cbc = cboost.CatBoostClassifier(iterations=2000,
                                             learning_rate=0.1,
                                             depth=8,
                                             eval_metric='Accuracy',
                                             random_seed=0,
                                             bagging_temperature=0.2,
                                             od_type='Iter',
                                             metric_period=75,
                                             od_wait=100)
                        # training the mew model
            self.cbc.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'CatBoost model train method done of the Model_Finder class')
            return self.cbc
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_CatBoost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'CatBoost Parameter tuning  failed. Exited the get_best_params_for_CatBoost method of the Model_Finder class')
            raise Exception()

    def get_best_model(self, train_x, train_y, test_x, test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        # create best model for KNN
        try:

            self.xgb_class = self.xgb_classifier(train_x, train_y)
            self.prediction_xgb_class = self.xgb_class.predict(test_x) # Predictions using the XGB Model
            self.prediction_xgb_auc = roc_auc_score(test_y,self.prediction_xgb_class)
            self.score.all_score(test_y,self.prediction_xgb_class,title="XGB Testing Score")
            self.prediction_xgb_class_train = self.xgb_class.predict(train_x)
            self.score.all_score(train_y, self.prediction_xgb_class_train, title="XGB Training Score")

            self.lgb_class = self.lgb_classifier(train_x, train_y)
            self.prediction_lgb_class = self.lgb_class.predict(test_x) # Predictions using the LGB Model
            self.prediction_lgb_auc = roc_auc_score(test_y,self.prediction_lgb_class)
            self.score.all_score(test_y,self.prediction_lgb_class,title="LGB Testing Score")
            self.prediction_lgb_class_train = self.lgb_class.predict(train_x)
            self.score.all_score(train_y, self.prediction_lgb_class_train, title="LGB Training Score")

            self.cb_class = self.catb_classifier(train_x, train_y)
            self.prediction_cb_class = self.cb_class.predict(test_x) # Predictions using the CatBoost Model
            self.prediction_cb_auc = roc_auc_score(test_y,self.prediction_cb_class)
            self.score.all_score(test_y,self.prediction_cb_class,title="Catboost Testing Score")
            self.prediction_cb_class_train = self.cb_class.predict(train_x)
            self.score.all_score(train_y, self.prediction_cb_class_train,title="Catboost Training Score")

            # #comparing the three models
            self.lst=[self.prediction_xgb_auc,self.prediction_lgb_auc,self.prediction_cb_auc]
            self.best_nm=np.argmax(self.lst)
            if self.best_nm==0:
                return 'XGBoost', self.xgb_class
            elif self.best_nm==1:
                return 'LGBoost',self.lgb_class
            else:
                return 'CatBoost',self.cb_class

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

