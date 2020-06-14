"""
This is the Entry point for Training the Machine Learning Model.

Version: 1.0
Revisions: None
"""
# Doing the necessary imports
from sklearn.model_selection import train_test_split
from data_ingestion import data_loader
from data_preprocessing import preprocessing
from data_preprocessing import clustering
from data_preprocessing_custom import preprocess_cus
from best_model_finder import tuner
from file_operations import file_methods
from application_logging import logger
# import numpy as np
# import pandas as pd
# import os

#Creating the common Logging object

class trainModel:

    def __init__(self):
        self.log_writer = logger.App_Logger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt", 'a+')

    def trainingModel(self):
        # Logging the start of Training
        self.log_writer.log(self.file_object, 'Start of Training')
        try:
            # Getting the data from the source
            data_getter=data_loader.Data_Getter(self.file_object,self.log_writer)
            data=data_getter.get_data()

            """doing the data preprocessing"""

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)
            #data.replace('?',np.NaN,inplace=True) # replacing '?' with NaN values for imputation

            # create separate features and labels
            X,Y=preprocessor.separate_label_feature(data,label_column_name='y')

            # Dropping column after performing EDA
            preprocessor_cus = preprocess_cus.Preprocessor_cus(self.file_object, self.log_writer)
            X=preprocessor_cus.drop_column(X)
            Y.replace({"no": 0, "yes": 1}, inplace=True) # Encoding Y (Predection label)

            # Response Encoding process
            # cat_cols2 = preprocessor_cus.categorical_column(X)
            # X = preprocessor_cus.ResponseEncoder(cat_cols2, X, Y)
            X = preprocessor_cus.test_data_encode(X) # Using Predefined Values

            print("Shape of the dataset after encoding: ", X.shape)

            # check if missing values are present in the dataset
            is_null_present,cols_with_missing_values=preprocessor.is_null_present(X)

            # if missing values are there, replace them appropriately.
            if(is_null_present):
                X=preprocessor.impute_missing_values(X) # missing value imputation


            features,label=preprocessor.handleImbalanceDataset(X,Y)
            # splitting the data into training and test set for each cluster one by one
            x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=1 / 5, random_state=42)

            model_finder=tuner.Model_Finder(self.file_object,self.log_writer) # object initialization

            #getting the best model for each of the clusters
            best_model_name,best_model=model_finder.get_best_model(x_train,y_train,x_test,y_test)

            #saving the best model to the directory.
            file_op = file_methods.File_Operation(self.file_object,self.log_writer)
            save_model=file_op.save_model(best_model,best_model_name)

            # logging the successful Training
            self.log_writer.log(self.file_object, 'Successful End of Training')
            self.file_object.close()

        except Exception as e:
            # logging the unsuccessful Training
            self.log_writer.log(self.file_object, 'Unsuccessful End of Training')
            self.file_object.close()
            raise Exception