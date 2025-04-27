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

    def __init__(self, dataset_folder, config: Configuration, feature_names, training):
        super().__init__(dataset_folder, config)

        self.x_test = None
        self.y_test = None
        self.y_test_strings = None
        self.num_test_instances = None
        self.training = training

        # total number of classes
        self.num_classes = None

        # dictionary with key: class as integer and value: array with index positions
        self.class_idx_to_ex_idxs_train = {}
        self.class_idx_to_ex_idxs_test = {}

        # np array that contains the number of instances for each classLabel in the training data
        self.num_instances_by_class_train = None

        # np array that contains the number of instances for each classLabel in the test data
        self.num_instances_by_class_test = None

        # np array that contains a list classes that occur in training OR test data set
        self.classes_total = feature_names

        # np array that contains a list classes that occur in training AND test data set
        self.classes_in_both = None

        # dictionary, key: class label, value: np array which contains 0s or 1s depending on whether the attribute
        # at this index is relevant for the class described with the label key
        self.class_label_to_masking_vector = {}

        self.group_id_to_masking_vector = {}

        #
        # new
        #

        self.y_train_strings_unique = None
        self.y_test_strings_unique = None

        # additional information for each example about their window time frame and failure occurrence time
        self.window_times_train = None
        self.window_times_test = None
        self.failure_times_train = None
        self.failure_times_test = None

        # numpy array (x,2) that contains each unique permutation between failure occurrence time and assigned label
        self.unique_failure_times_label = None
        self.failure_times_count = None

        # pandas df ( = matrix) with pair-wise similarities between labels in respect to a metric
        self.df_label_sim_localization = None
        self.df_label_sim_failuremode = None
        self.df_label_sim_condition = None

    #if useNPYCaseBase then casebase param is empty
    def load_files(self, queries, caseBase):



        self.x_train = np.array(caseBase['timeseries_array'])
        self.y_train_strings = np.array(caseBase['labels']).reshape(-1, 1)
        self.window_times_train = np.expand_dims(np.array(caseBase['window_times']), axis=-1)
        self.failure_times_train = np.array(caseBase['recording_sequences']).reshape(-1, 1)


        self.x_test = np.expand_dims(np.array(queries['timeseries_array']), axis=0)
        self.y_test_strings = np.array([queries['label']]).reshape(-1, 1)
        self.window_times_test = np.expand_dims(np.array(queries['window_time']).reshape(-1, 1), axis=0)
        self.failure_times_test = np.array([queries['recording_sequences']]).reshape(-1, 1)


        print(self.y_test_strings.shape)
        print(self.window_times_test.shape)
        print(self.failure_times_test.shape)
        self.feature_names_all = np.load(self.dataset_folder + 'feature_names.npy')  # names of the features (3. dim)

    def load(self, queries, caseBase, print_info=True):
        self.load_files(queries, caseBase)

        # create a encoder, sparse output must be disabled to get the intended output format
        # added categories='auto' to use future behavior
        one_hot_encoder = preprocessing.OneHotEncoder(sparse=False, categories='auto')

        # prepare the encoder with training and test labels to ensure all are present
        # the fit-function 'learns' the encoding but does not jet transform the data
        # the axis argument specifies on which the two arrays are joined
        one_hot_encoder = one_hot_encoder.fit(np.concatenate((self.y_train_strings, self.y_test_strings), axis=0))

        # transforms the vector of labels into a one hot matrix
        self.y_train = one_hot_encoder.transform(self.y_train_strings)
        self.y_test = one_hot_encoder.transform(self.y_test_strings)

        # reduce to 1d array
        self.y_train_strings = np.squeeze(self.y_train_strings)
        self.y_test_strings = np.squeeze(self.y_test_strings, axis=1)

        ##
        # safe information about the dataset
        ##

        # length of the first array dimension is the number of examples
        self.num_train_instances = self.x_train.shape[0]
        self.num_test_instances = self.x_test.shape[0]

        # the total sum of examples
        self.num_instances = self.num_train_instances + self.num_test_instances

        # length of the second array dimension is the length of the time series
        self.time_series_length = self.x_train.shape[1]

        # length of the third array dimension is the number of channels = (independent) readings at this point of time
        self.time_series_depth = self.x_train.shape[2]

        self.y_test_strings_unique, counts = np.unique(self.y_test_strings, return_counts=True)
        self.num_instances_by_class_test = np.asarray((self.y_test_strings_unique, counts)).T

        # required for inference metric calculation
        # get all failures and labels as unique entry
        failure_times_label = np.stack((self.y_test_strings, np.squeeze(self.failure_times_test, axis=1))).T
        # extract unique permutations between failure occurrence time and labeled entry
        unique_failure_times_label, failure_times_count = np.unique(failure_times_label, axis=0, return_counts=True)
        # remove noFailure entries
        idx = np.where(np.char.find(unique_failure_times_label, 'noFailure') >= 0)
        self.unique_failure_times_label = np.delete(unique_failure_times_label, idx, 0)
        self.failure_times_count = np.delete(failure_times_count, idx, 0)

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

    def get_sim_label_pair_for_notion(self, label_1: str, label_2: str, notion_of_sim: str):
        # Output similarity value under consideration of the metric

        if notion_of_sim == 'failuremode':
            pair_label_sim = self.df_label_sim_failuremode.loc[label_1, label_2]
        elif notion_of_sim == 'localization':
            pair_label_sim = self.df_label_sim_localization.loc[label_1, label_2]
        elif notion_of_sim == 'condition':
            pair_label_sim = self.df_label_sim_condition.loc[label_1, label_2]
        else:
            print("Similarity notion: ", notion_of_sim, " unknown! Results in sim 0")
            pair_label_sim = 0

        return float(pair_label_sim)
