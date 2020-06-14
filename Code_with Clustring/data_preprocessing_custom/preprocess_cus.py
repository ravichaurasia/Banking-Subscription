import pandas as pd
import numpy as np
from sklearn.utils import resample

import json
import os
class Preprocessor_cus:
    """

        Class Name: Preprocessor_cus
        Description: This class shall  be used to clean and transform the data before training.
        Output: preprocessed output
        On Failure: Raise Exception

        Version: 1.0
        Revisions: None
        """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.path = os.getcwd() + '/data_preprocessing_custom/tst_encode.json'

    def drop_column(self,X):
        """
            Method Name: drop_column
            Description: This method finds out the specific columns and droped them based upon performed EDA.
            Output: DataFrame with two droped column
            On Failure: Raise Exception

            Version: 1.0
            Revisions: None
            """
        try:
            self.X = X.drop(['duration', 'day_of_week'], axis=1)
            for cl in self.X:
                if cl =='Unnamed: 0':
                    self.X = X.drop(['Unnamed: 0'], axis=1)
            return self.X
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in drop_column method of the Preprocessor custom class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,
                                   'Column drop Unsuccessful. Exited the drop_column method of the Preprocessor custom class')
            raise Exception()

    def up_sample(self,X, y):
        """It takes the X and y data frames,
        In output it provides upsampled data frames"""

        # concatenate our training data back together
        self.X4 = pd.concat([X, y], axis=1)
        try:
            # separate minority and majority classes

            self.no = self.X4[self.X4.y == 0]
            self.yes = self.X4[self.X4.y == 1]
            # print('yes ,no:' ,yes.count(), no.count())
            a1 = self.no.count()[0]
            a2 = self.yes.count()[0]
            if a1 >= a2:
                # upsample minority
                upsampled = resample(self.yes,
                                     replace=True,  # sample with replacement
                                     n_samples=len(self.no),  # match number in majority class
                                     random_state=27)  # reproducible results
                upsampled = pd.concat([self.no, upsampled])
            else:
                # upsample minority
                upsampled = resample(self.no,
                                     replace=True,  # sample with replacement
                                     n_samples=len(self.yes),  # match number in majority class
                                     random_state=27)  # reproducible results
                # combine majority and upsampled minority
                upsampled = pd.concat([self.yes, upsampled])

            # check new class counts
            # upsampled.y.value_counts()
            upsampled = upsampled.reset_index().drop("index", axis=1)
            return upsampled
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in up_sample method of the Preprocessor custom class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,
                                   'up sample Unsuccessful. Exited the up_sample method of the Preprocessor custom class')
            raise Exception()

    def categorical_column(self,data):
        """It takes the data frame and return the list of categorical column"""
        try:
            # Categorical boolean mask
            categorical_feature_mask = data.dtypes == object

            # filter categorical columns by using mask and turn it into a list
            categorical_cols = data.columns[categorical_feature_mask].tolist()
            return categorical_cols
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in categorical_column method of the Preprocessor custom class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,'categorical columne Unsuccessful. Exited the categorical_column method of the Preprocessor custom class')
            raise Exception()

    def numerical_column(self,data):
        """It takes the data frame and return the list of numerical column
        Note:-here it takes only dtypes == int64 and float64,
        which can be modified as per requirement"""
        try:
            # filter categorical columns by using mask and turn it into a list
            num_cols1 = data.columns[data.dtypes == 'int64'].tolist()
            num_cols2 = data.columns[data.dtypes == 'float64'].tolist()
            num_cols1.extend(num_cols2)
            return num_cols1
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in numerical_column method of the Preprocessor custom class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,'numerical columne Unsuccessful. Exited the numerical_column method of the Preprocessor custom class')
            raise Exception()



    def feature_dict(self,alpha, feature, train_df, train_df_y):
        # value_count: it contains a dict like
        try:
            print('X feature :-', feature)
            value_count = train_df[feature].value_counts()

            # feat_dict : Categorical feature Dict, which contains the probability array for each categorical variable
            feat_dict = dict()

            # denominator will contain the number of time that particular feature occured in whole data
            for i, denominator in value_count.items():
                # vec will contain (p(yi==1/Gi) probability of the particular
                # categorical feature belongs to particular class
                # vec is 2 diamensional vector
                vec = []
                for k in range(0, 2):
                    # cls_cnt.shape[0] will return the number of rows

                    cls_cnt = train_df.loc[(train_df_y == k) & (train_df[feature] == i)]

                    # cls_cnt.shape[0](numerator) will contain the number of time that particular feature occured in whole data
                    vec.append((cls_cnt.shape[0] + alpha * 10) / (denominator + 20 * alpha))

                # we are adding the categorical feature to the dict as key and vec as value
                feat_dict[i] = vec
            return feat_dict
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in feature_dict method of the Preprocessor custom class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,'feature dict Unsuccessful. Exited the feature_dict method of the Preprocessor custom class')
            raise Exception()


    def response_feature(self,alpha, feature, train_df, train_df_y):
        # Get Response coded feature
        try:
            feat_dict = self.feature_dict(alpha, feature, train_df, train_df_y)
            # value_count is similar in get_fea_dict
            value_count = train_df[feature].value_counts()

            # res_fea: response coded feature, it will contain the response coded feature for each feature value in the data
            res_fea = []
            # for every feature values in the given data frame we will check if it is there in the train data
            # then we will add the feature to response_feature
            # if not we will add [1/2, 1/2] to response_feature
            for index, row in train_df.iterrows():
                if row[feature] in dict(value_count).keys():
                    res_fea.append(feat_dict[row[feature]])
                else:
                    res_fea.append([1 / 2, 1 / 2])
            return res_fea
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in response_feature method of the Preprocessor custom class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,'response feature Unsuccessful. Exited the response_feature method of the Preprocessor custom class')
            raise Exception()


    def ResponseEncoder(self,categorical_cols, x_df, y_df):
        """
        This function takes Categorical column names and X and Y dataframe.

        Returns the response coded dataframe
        """
        try:
            print("Encoding dataset")
            print("Shape of the dataset before encoding: ", x_df.shape)
            for i in (categorical_cols):
                temp_response_coded_feature = np.array(self.response_feature(1, i, x_df, y_df))
                df_response = pd.DataFrame(temp_response_coded_feature, columns=[i + "_0", i + "_1"])
                x_df = pd.concat([x_df, df_response], axis=1)

            # Remove the categorical features as the response coded features are added
            x_df = x_df.drop(categorical_cols, axis=1)
            x_df.to_csv(os.getcwd() + "/preprocessing_data/" + 'encoded_inputFile.csv', index=None, header=True) # Save encoded file at preprocessing_data folder
            x_df = pd.read_csv(os.getcwd() + "/preprocessing_data/" + 'encoded_inputFile.csv') # Open encoded file from preprocessing_data folder
            return x_df
        except Exception as e:
            self.logger_object.log(self.file_object,'Exception occured in ResponseEncoder method of the Preprocessor custom class. Exception message:  '+str(e))
            self.logger_object.log(self.file_object,'Response Encoder Unsuccessful. Exited the ResponseEncoder method of the Preprocessor custom class')
            raise Exception()

    def test_data_encode(self, X):
        """
        It can return directly with saved values from Response Encoder
        and decrease the processing latency
        """
        try:

            aa = pd.read_json(self.path)
            n_col = self.numerical_column(X)
            n_X = X[n_col]
            for i, j in enumerate(aa):
                #     print(j)
                col0 = X[j].replace(aa[j][0])
                col1 = X[j].replace(aa[j][1])
                col0, col1 = pd.DataFrame(col0), pd.DataFrame(col1)
                col = pd.concat([col0, col1], axis=1)
                col.columns = [(str(j) + '_0'), (str(j) + '_1')]
                n_X = pd.concat([n_X, col], axis=1)
            return n_X
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in test_data_encode method of the Preprocessor custom class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'test-data encode Unsuccessful. Exited the test_data_encode method of the Preprocessor custom class')
            raise Exception()