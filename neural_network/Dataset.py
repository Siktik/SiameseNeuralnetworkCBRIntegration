from time import perf_counter

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
        self.one_hot_encoder_labels = None  # one hot label encoder
        self.classes_Unique_oneHotEnc = None
        self.num_train_instances = None
        self.num_instances = None

        # Class names as string
        self.classes_total = None

        self.time_series_length = None
        self.time_series_depth = None

        # the names of all features of the dataset loaded from files
        self.feature_names_all = None

        self.x_train_TSFresh_features = None
        self.x_test_TSFresh_features = None
        self.relevant_features_by_TSFresh = None

    def load(self):
        raise NotImplemented('Not implemented for abstract class')

    @staticmethod
    def draw_from_ds(self, dataset, num_instances, is_positive, class_idx=None):
        # dataset: vector with one-hot encoded label of the data set

        # draw as long as is_positive criterion is not satisfied

        # draw two random examples index
        if class_idx is None:
            while True:
                first_idx = np.random.randint(0, num_instances, size=1)[0]
                second_idx = np.random.randint(0, num_instances, size=1)[0]
                # return the two indexes if they match the is_positive criterion
                if is_positive:
                    if np.array_equal(dataset[first_idx], dataset[second_idx]):
                        return first_idx, second_idx
                else:
                    if not np.array_equal(dataset[first_idx], dataset[second_idx]):
                        return first_idx, second_idx
        else:
            # examples are drawn by a given class index
            # contains idx values of examples from the given class
            class_idx_arr = self.class_idx_to_ex_idxs_train[class_idx]

            # print("class_idx:", class_idx, " class_idx_arr: ", class_idx_arr, "self.class_idx_to_class_string: ",
            #      self.class_idx_to_class_string[class_idx])

            # Get a random idx of an example that is part of this class
            first_rand_idx = np.random.randint(0, len(class_idx_arr), size=1)[0]
            first_idx = class_idx_arr[first_rand_idx]

            if is_positive:
                while True:
                    second_rand_idx = np.random.randint(0, len(class_idx_arr), size=1)[0]
                    second_idx = class_idx_arr[second_rand_idx]
                    if first_idx != second_idx:
                        return first_idx[0], second_idx[0]
            else:
                while True:
                    uniform_sampled_class = np.random.randint(low=0,
                                                              high=len(self.y_train_strings_unique),
                                                              size=1)
                    class_idx_arr_neg = self.class_idx_to_ex_idxs_train[uniform_sampled_class[0]]
                    second_rand_idx_neg = np.random.randint(0, len(class_idx_arr_neg), size=1)[0]
                    # print("uniform_sampled_class: ", uniform_sampled_class, "class_idx_arr_neg: ", class_idx_arr_neg,
                    #       "second_rand_idx_neg: ", second_rand_idx_neg)

                    second_idx = class_idx_arr_neg[second_rand_idx_neg]
                    # second_idx = np.random.randint(0, num_instances, size=1)[0]

                    if second_idx not in class_idx_arr[:, 0]:
                        # print("class_idx_arr: ", class_idx_arr, " - uniform_sampled_class: ",
                        # uniform_sampled_class[0])
                        return first_idx[0], second_idx[0]


class FullDataset(Dataset):

    def __init__(self, dataset_folder, config: Configuration, training):
        super().__init__(dataset_folder, config)

        self.x_test = None
        self.y_test = None
        self.y_test_strings = None
        self.num_test_instances = None
        self.training = training

        self.x_train_TSFresh_features = None
        self.x_test_TSFresh_features = None
        self.relevant_features_by_TSFresh = None

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
        self.classes_total = None

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

    def load_files(self, features, labels, windowTimes, recordingSequence):

        self.x_train = np.load(self.dataset_folder + 'train_features.npy')  # data training
        self.y_train_strings = np.expand_dims(np.load(self.dataset_folder + 'train_labels.npy'), axis=-1)
        self.window_times_train = np.expand_dims(np.load(self.dataset_folder + 'train_window_times.npy'), axis=-1)
        self.failure_times_train = np.expand_dims(np.load(self.dataset_folder + 'train_failure_times.npy'), axis=-1)


        self.x_test = np.load(self.dataset_folder + 'test_features.npy')  # data testing
        self.y_test_strings = np.expand_dims(np.load(self.dataset_folder + 'test_labels.npy'), axis=-1)
        self.window_times_test = np.expand_dims(np.load(self.dataset_folder + 'test_window_times.npy'), axis=-1)
        self.failure_times_test = np.expand_dims(np.load(self.dataset_folder + 'test_failure_times.npy'), axis=-1)
        
        '''
        self.x_test= features
        self.y_test_strings = labels
        self.window_times_test = np.expand_dims(windowTimes, axis=-1)
        self.failure_times_test = recordingSequence
        '''
        print(windowTimes)
        print(self.y_test_strings.shape)
        print(self.window_times_test.shape)
        print(self.failure_times_test.shape)
        self.feature_names_all = np.load(self.dataset_folder + 'feature_names.npy')  # names of the features (3. dim)

    def load(self, features, labels, windowTimes, recordingSequence, print_info=True):
        self.load_files(features, labels, windowTimes, recordingSequence)

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
        self.y_test_strings = np.squeeze(self.y_test_strings)

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

        # get the unique classes and the corresponding number
        self.classes_total = np.unique(np.concatenate((self.y_train_strings, self.y_test_strings), axis=0))
        self.classes_Unique_oneHotEnc = one_hot_encoder.transform(np.expand_dims(self.classes_total, axis=1))
        self.num_classes = self.classes_total.size

        # Create two dictionaries to link/associate each class with all its training examples
        for i in range(self.num_classes):
            self.class_idx_to_ex_idxs_train[i] = np.argwhere(self.y_train[:, i] > 0)
            self.class_idx_to_ex_idxs_test[i] = np.argwhere(self.y_test[:, i] > 0)

        # collect number of instances for each class in training and test
        self.y_train_strings_unique, counts = np.unique(self.y_train_strings, return_counts=True)
        self.num_instances_by_class_train = np.asarray((self.y_train_strings_unique, counts)).T
        self.y_test_strings_unique, counts = np.unique(self.y_test_strings, return_counts=True)
        self.num_instances_by_class_test = np.asarray((self.y_test_strings_unique, counts)).T

        # calculate the number of classes that are the same in test and train
        self.classes_in_both = np.intersect1d(self.num_instances_by_class_test[:, 0],
                                              self.num_instances_by_class_train[:, 0])

        # required for inference metric calculation
        # get all failures and labels as unique entry
        failure_times_label = np.stack((self.y_test_strings, np.squeeze(self.failure_times_test))).T
        # extract unique permutations between failure occurrence time and labeled entry
        unique_failure_times_label, failure_times_count = np.unique(failure_times_label, axis=0, return_counts=True)
        # remove noFailure entries
        idx = np.where(np.char.find(unique_failure_times_label, 'noFailure') >= 0)
        self.unique_failure_times_label = np.delete(unique_failure_times_label, idx, 0)
        self.failure_times_count = np.delete(failure_times_count, idx, 0)

        self.calculate_maskings()
        self.load_sim_matrices()

        # data
        # 1. dimension: example
        # 2. dimension: time index
        # 3. dimension: array of all channels

        if print_info:
            print()
            print('Dataset loaded:')
            print('Shape of training set (example, time, channels):', self.x_train.shape)
            print('Shape of test set (example, time, channels):', self.x_test.shape)
            print('Num of classes in train and test together:', self.num_classes)
            # print('Classes used in training: ', len(self.y_train_strings_unique)," :",self.y_train_strings_unique)
            # print('Classes used in test: ', len(self.y_test_strings_unique)," :", self.y_test_strings_unique)
            # print('Classes in total: ', self.classes_total)
            print()


    def load_sim_matrices(self):
        # load a matrix with pair-wise similarities between labels in respect
        # to different metrics
        self.df_label_sim_failuremode = pd.read_csv(self.dataset_folder + 'FailureMode_Sim_Matrix.csv', sep=';',
                                                    index_col=0)
        self.df_label_sim_failuremode.index = self.df_label_sim_failuremode.index.str.replace('\'', '')
        self.df_label_sim_localization = pd.read_csv(self.dataset_folder + 'Localization_Sim_Matrix.csv', sep=';',
                                                     index_col=0)
        self.df_label_sim_localization.index = self.df_label_sim_localization.index.str.replace('\'', '')
        self.df_label_sim_condition = pd.read_csv(self.dataset_folder + 'Condition_Sim_Matrix.csv', sep=';',
                                                  index_col=0)
        self.df_label_sim_condition.index = self.df_label_sim_condition.index.str.replace('\'', '')

    def calculate_maskings(self):
        for case in self.classes_total:

            if self.config.individual_relevant_feature_selection:
                relevant_features_for_case = self.config.get_relevant_features_case(case)
            else:
                relevant_features_for_case = self.config.get_relevant_features_group(case)

            if self.config.use_additional_strict_masking_for_attribute_sim:
                masking1 = np.isin(self.feature_names_all, relevant_features_for_case[0])
                masking2 = np.isin(self.feature_names_all, relevant_features_for_case[1])
                self.class_label_to_masking_vector[case] = [masking1, masking2]
            else:
                masking = np.isin(self.feature_names_all, relevant_features_for_case)
                self.class_label_to_masking_vector[case] = masking

        for group_id, features in self.config.group_id_to_features.items():
            masking = np.isin(self.feature_names_all, features)
            self.group_id_to_masking_vector[group_id] = masking

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


