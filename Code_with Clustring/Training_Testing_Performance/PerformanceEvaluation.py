from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import os
sns.set()

class performance:
    """
    It takes the input of true and predicted value and
    provides the evaluation graph in Training_Testing_Performance/Plots folder
    """
    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def all_score(self,true, predictions,cls,title):
        """
                        Method Name: all_score
                        Description: This method takes the true value and predicted values
                        Output: It provides graph.
                        On Failure: Raise Exception

                        Version: 1.0
                        Revisions: None

                """
        try:
            plt.clf()
            f1score = metrics.f1_score(true, predictions)
            precision = metrics.precision_score(true, predictions)
            recall = metrics.recall_score(true, predictions)
            auc = metrics.roc_auc_score(true, predictions)
            pl2 = [f1score, precision, recall, auc]
            xx = ['f1score', 'precision', 'recall', 'auc']

            ax = sns.barplot(xx, pl2)
            for p in ax.patches:
                ax.annotate('{:.2f}'.format(p.get_height()), (p.get_x() + 0.25, p.get_height() + 0.004))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=22, ha="right")
            ax.set_ylim(0, 1)
            plt.ylabel("Score")
            plt.title(str(title))

            fig = plt.gcf()
            # plt.show()
            # plt.draw()
            self.pth=os.getcwd()+"/Training_Testing_Performance/Cluster"+str(cls)
            # Making Directory
            if not os.path.isdir(self.pth):
                os.makedirs(self.pth)
            fig.savefig(self.pth+"/"+title+'-Plot.PNG', dpi=150)  # saving the Performance plot locally
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in all_score method of the Performance class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'all score graph Unsuccessful. Exited the all_score method of the Performance class')
            raise Exception()