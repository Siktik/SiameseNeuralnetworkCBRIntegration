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
        self.y_test_strings = None
        self.training = training

        # all failure names
        self.classes_total = failure_names

        # dictionary, key: class label, value: np array which contains 0s or 1s depending on whether the attribute
        # at this index is relevant for the class described with the label key
        self.class_label_to_masking_vector = {}

        # additional information for each example about their window time frame and failure occurrence time
        self.window_times_train = None
        self.window_times_test = None
        self.failure_times_train = None
        self.failure_times_test = None

    def update_query(self, queries):

        # only query is updated
        self.x_test = np.expand_dims(np.array(queries['timeseries_array']), axis=0)
        self.y_test_strings = np.array([queries['label']]).reshape(-1, 1)
        self.window_times_test = np.expand_dims(np.array(queries['window_time']).reshape(-1, 1), axis=0)
        self.failure_times_test = np.array([queries['recording_sequences']]).reshape(-1, 1)

    def load_files(self, queries, caseBase):

        if queries is None and caseBase is None:
            self.x_train = np.load(self.dataset_folder + 'train_features.npy')  # data training
            self.y_train_strings = np.expand_dims(np.load(self.dataset_folder + 'train_labels.npy'), axis=-1)
            self.window_times_train = np.expand_dims(np.load(self.dataset_folder + 'train_window_times.npy'), axis=-1)
            self.failure_times_train = np.expand_dims(np.load(self.dataset_folder + 'train_failure_times.npy'), axis=-1)

            self.x_test = np.load(self.dataset_folder + 'test_features.npy')  # data testing
            self.y_test_strings = np.expand_dims(np.load(self.dataset_folder + 'test_labels.npy'), axis=-1)
            self.window_times_test = np.expand_dims(np.load(self.dataset_folder + 'test_window_times.npy'), axis=-1)
            self.failure_times_test = np.expand_dims(np.load(self.dataset_folder + 'test_failure_times.npy'), axis=-1)
        else:
            self.x_train = np.array(caseBase['timeseries_array'])
            self.y_train_strings = np.array(caseBase['labels']).reshape(-1, 1)
            self.window_times_train = np.expand_dims(np.array(caseBase['window_times']), axis=-1)
            self.failure_times_train = np.array(caseBase['recording_sequences']).reshape(-1, 1)

            self.update_query(queries)

        self.feature_names_all = np.load(self.dataset_folder + 'feature_names.npy')  # names of the features (3. dim)

    def load(self, queries, caseBase, print_info=True):
        self.load_files(queries, caseBase)

        # reduce to 1d array
        self.y_train_strings = np.squeeze(self.y_train_strings)
        self.y_test_strings = np.squeeze(self.y_test_strings, axis=1)

        # length of the second array dimension is the length of the time series
        self.time_series_length = self.x_train.shape[1]

        # length of the third array dimension is the number of channels = (independent) readings at this point of time
        self.time_series_depth = self.x_train.shape[2]

        self.calculate_maskings()

        # data
        # 1. dimension: example
        # 2. dimension: time index
        # 3. dimension: array of all channels

        if print_info:
            print()
            print('Dataset loaded:')
            print('Shape of training set (example, time, channels):', self.x_train.shape)
            print('Shape of test set (example, time, channels):', self.x_test.shape)
            # print('Classes used in training: ', len(self.y_train_strings_unique)," :",self.y_train_strings_unique)
            # print('Classes used in test: ', len(self.y_test_strings_unique)," :", self.y_test_strings_unique)
            # print('Classes in total: ', self.classes_total)
            print()

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

    def get_time_window_str(self, index, dataset_type):
        if dataset_type == 'test':
            dataset = self.window_times_test
        elif dataset_type == 'train':
            dataset = self.window_times_train
        else:
            raise ValueError('Unknown dataset type')

        rep = lambda x: str(x).replace("['YYYYMMDD HH:mm:ss (", "").replace(")']", "")

        t1 = rep(dataset[index][0])
        t2 = rep(dataset[index][2])
        return " - ".join([t1, t2])

    def get_indices_failures_only_test(self):
        return np.where(self.y_test_strings != 'no_failure')[0]


