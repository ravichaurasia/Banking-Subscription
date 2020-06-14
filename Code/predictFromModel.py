import pandas
from file_operations import file_methods
from data_preprocessing import preprocessing
from data_ingestion import data_loader_prediction
from application_logging import logger
from Prediction_Raw_Data_Validation.predictionDataValidation import Prediction_Data_validation
# import pickle
from data_preprocessing_custom import preprocess_cus
import os

class prediction:

    def __init__(self,path):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.pred_data_val = Prediction_Data_validation(path)

    def predictionFromModel(self):

        try:
            self.pred_data_val.deletePredictionFile() #deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object,'Start of Prediction')
            data_getter=data_loader_prediction.Data_Getter_Pred(self.file_object,self.log_writer)
            data=data_getter.get_data()

            #cdrop 'Unnamed: 0'
            for cl in data.columns:
                if cl == 'Unnamed: 0':
                    data.drop('Unnamed: 0', axis=1, inplace=True)

            # Dropping column after performing EDA
            preprocessor_cus = preprocess_cus.Preprocessor_cus(self.file_object, self.log_writer)
            data=preprocessor_cus.drop_column(data)

            preprocessor=preprocessing.Preprocessor(self.file_object,self.log_writer)

            # replacing '?' values with np.nan as discussed in the EDA part
            data = preprocessor.replaceInvalidValuesWithNull(data)

            # get encoded values for categorical data
            data = preprocessor_cus.test_data_encode(data)

            is_null_present,cols_with_missing_values=preprocessor.is_null_present(data)
            if(is_null_present):
                data=preprocessor.impute_missing_values(data)

            #data=data.to_numpy()
            file_loader=file_methods.File_Operation(self.file_object,self.log_writer)

            result=[] # initialize balnk list for storing predicitons

            model = file_loader.load_model('CatBoost')
            for val in (model.predict(data)):
                result.append(val)

            result = pandas.DataFrame(result,columns=['Predictions'])
            path="Prediction_Output_File/Predictions.csv"
            result['Predictions'].replace({ 0:"no",  1:"yes"}, inplace=True)
            result.to_csv("Prediction_Output_File/Predictions.csv",header=True) #appends result to prediction file
            self.log_writer.log(self.file_object,'End of Prediction')
        except Exception as ex:
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return path




