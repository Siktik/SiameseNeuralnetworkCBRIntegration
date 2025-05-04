import numpy as np
import pandas as pd
from sklearn import preprocessing

from configuration.Configuration import Configuration


class Dataset:

    def __init__(self, dataset_folder, config: Configuration):
        self.dataset_folder = dataset_folder
        self.config: Configuration = config

        self.x_train = None  # training data (examples,time,channels)
        self.y_train = None  # One hot encoded class labels (numExamples,numClasses)
        self.y_train_strings = None  # class labels as strings (numExamples,1)
        self.num_train_instances = None
        self.num_instances = None

        # Class names as string
        self.classes_total = None

        self.time_series_length = None
        self.time_series_depth = None

        # the names of all features of the dataset loaded from files
        self.feature_names_all = None

    def load(self, queries, casebase):
        raise NotImplemented('Not implemented for abstract class')


class FullDataset(Dataset):

    def __init__(self, dataset_folder, config: Configuration, failure_names, training):
        super().__init__(dataset_folder, config)

        self.x_test = None
        self.y_test = None
        self.y_train_strings = None
        self.training = training

        # all failure names
        self.classes_total = failure_names

        # dictionary, key: class label, value: np array which contains 0s or 1s depending on whether the attribute
        # at this index is relevant for the class described with the label key
        self.class_label_to_masking_vector = {}

    def update_query(self, queries):

        # only query is updated
        self.x_test = np.expand_dims(np.array(queries['timeseries_array']), axis=0)

    def load_files(self, queries, caseBase):

        self.x_train = np.array(caseBase['timeseries_array'])
        self.y_train_strings = np.array(caseBase['labels']).reshape(-1, 1)
        self.update_query(queries)
        self.feature_names_all = np.load(self.dataset_folder + 'feature_names.npy')  # names of the features (3. dim)

    def load(self, queries, caseBase, print_info=True):
        self.load_files(queries, caseBase)

        # reduce to 1d array
        self.y_train_strings = np.squeeze(self.y_train_strings)

        # length of the second array dimension is the length of the time series
        self.time_series_length = self.x_train.shape[1]

        # length of the third array dimension is the number of channels = (independent) readings at this point of time
        self.time_series_depth = self.x_train.shape[2]

        self.calculate_maskings()


    def calculate_maskings(self):
        for case in self.classes_total:
            relevant_features_for_case = self.config.get_relevant_features_case(case)

            masking1 = np.isin(self.feature_names_all, relevant_features_for_case[0])
            masking2 = np.isin(self.feature_names_all, relevant_features_for_case[1])
            self.class_label_to_masking_vector[case] = [masking1, masking2]

    # returns a boolean array with values depending on whether the attribute at this index is relevant
    # for the class of the passed label
    def get_masking(self, class_label):

        if class_label not in self.class_label_to_masking_vector:
            raise ValueError('Passed class label', class_label, 'was not found in masking dictionary')
        else:
            if self.config.use_additional_strict_masking_for_attribute_sim:
                masking = self.class_label_to_masking_vector.get(class_label)
                masking_vec = np.concatenate((masking[0], masking[1]))
                return masking_vec
            else:
                return self.class_label_to_masking_vector.get(class_label)

    def get_masking_float(self, class_label):
        return self.get_masking(class_label).astype(float)



