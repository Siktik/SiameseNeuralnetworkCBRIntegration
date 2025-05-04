import numpy as np
import os
from configuration.Configuration import Configuration


class Dataset:

    def __init__(self, config: Configuration, failure_names, training):

        self.x_train = None
        self.config: Configuration = config
        self.feature_names_all = None
        self.x_test = None
        self.y_test = None
        self.y_train_strings = None
        self.training = training
        self.time_series_length = None
        self.time_series_depth = None

        # all failure names
        self.classes_total = failure_names

        # dictionary, key: class label, value: np array which contains 0s or 1s depending on whether the attribute
        # at this index is relevant for the class described with the label key
        self.class_label_to_masking_vector = {}

    def update_query(self, queries):

        # only query is updated
        self.x_test = np.expand_dims(np.array(queries['timeseries_array']), axis=0)

    def load(self, queries, case_base):

        self.x_train = np.array(case_base['timeseries_array'])
        self.y_train_strings = np.array(case_base['labels']).reshape(-1, 1)
        self.update_query(queries)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        feature_names_path = os.path.join(current_dir, '..', 'configuration', 'feature_names.npy')
        self.feature_names_all = np.load(feature_names_path)  # names of the features (3. dim)


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



