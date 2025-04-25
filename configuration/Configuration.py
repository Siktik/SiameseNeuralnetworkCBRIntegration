import json


####
# Note: Division into different classes only serves to improve clarity.
# Only the Configuration class should be used to access all variables.
# Important: It must be ensured that variable names are only used once.
# Otherwise they will be overwritten depending on the order of inheritance!
# All methods should be added to the Configuration class to be able to access all variables
####

class GeneralConfiguration:

    def __init__(self):
        ###
        # This configuration contains overall settings that couldn't be match to a specific program component
        ###

        # Specifies the maximum number of gpus used
        self.max_gpus_used = 4

        # Specifies the maximum number of cores to be used
        self.max_parallel_cores = 60

        # Folder where the trained models are saved to during learning process
        self.models_folder = '../data/trained_models/'

        # Path and file name to the specific model that should be used for testing and live classification
        self.filename_model_to_use = 'temp_snn_model_04-09_22-31-47_epoch-1748/'
        self.directory_model_to_use = self.models_folder + self.filename_model_to_use + '/'

        self.number_of_subsequent_retrievals = 1


class ModelConfiguration:

    def __init__(self):
        pass

        ###
        # This configuration contains all parameters defining the structure of the classifier.
        # (SNNs as well as the CBS similarity measure)
        ###

        ##
        # Architecture (independent of whether a single SNN or the CBS is used)
        ##

        # standard = classic snn behaviour, context vectors calculated each time, also multiple times for the example
        # fast = encoding of case base only once, example also only once
        # ffnn = uses ffnn as distance measure
        # simple = mean absolute difference as distance measure instead of the ffnn

        # following two lines were set in Kleins Code
        # self.architecture_variants = ['standard_simple', 'standard_ffnn', 'fast_simple', 'fast_ffnn']
        # self.architecture_variant = self.architecture_variants[0]

        # based on the paper by Klein the final SNN that performed best is the standard_simple with an abs_mean sim measure
        # that abs_mean was used can be found in the paper. That standard_simple was used is not directly mentioned but
        # by description of the SNN in the paper and the fact it is selected in the final vaersion of the accompanying code
        # suggests this is the SNN config leading to the SNN that performed best
        self.architecture_variant = 'standard_simple'

        ##
        # Determines how the similarity between two embedding vectors is determined (when a simple architecture is used)
        ##

        # following two lines were set in Kleins Code
        # self.simple_measures = ['abs_mean', 'euclidean_sim', 'euclidean_dis', 'dot_product', 'cosine']
        # self.simple_measure = self.simple_measures[0]
        self.simple_measure = 'abs_mean'

        ###
        # Hyperparameters
        ###

        # Main directory where the hyperparameter config files are stored
        self.hyper_file_folder = '../configuration/hyperparameter_combinations/'
        self.use_hyper_file = True

        # holding the hyperparameters for the cnn2d_withAddInput encoder variant
        self.hyper_file = self.hyper_file_folder + 'cnn2d_withAddInput'

        ##
        # Various settings influencing the similarity calculation
        ##

        # SNN output is normalized (x = x/|x|) (useful for eucl.?)
        self.normalize_snn_encoder_output = False  # default: False

        # Additional option for encoder variant cnn2dwithaddinput and the euclidean distance:
        # Weighted euclidean similarity based on relevant attributes
        self.useFeatureWeightedSimilarity = True  # default: False

        # Weights are based on masking vectors that contain 1 if a feature is selected as relevant for a
        # label (failure mode) and 0 otherwise. If option is set False then features based
        # on groups are used.

        # Select whether the reduction to relevant features should be based on the case itself or the group it belongs
        # to. Based on case = True, based on group = False
        # Must be false for CBS!
        self.individual_relevant_feature_selection = True  # default: True

        # Using the more restrictive features as additional masking vector for feature sim calculation
        # in cnn_with_add_input
        self.use_additional_strict_masking_for_attribute_sim = True  # default: False

        # Option to simulate a retrieval situation (during training) where only the weights of the
        # example from the case base/training data set are known:
        self.use_same_feature_weights_for_unsimilar_pairs = True  # default: True

        # Compares each time step of the encoded representation with each other time step
        # (instead of only comparing the ones with the same indices)
        # Implementation is based on NeuralWarp FFNN but used for simple similarity measures
        self.use_time_step_wise_simple_similarity = False  # default: False


class InferenceConfiguration:

    def __init__(self):
        ##
        # Settings and parameters for all inference processes
        ##
        # Notes:
        #   - Folder of used model is specified in GeneralConfiguration
        #   - Does not include settings for BaselineTester


        # Parameter to control the size / number of the queries used for evaluation
        self.inference_with_failures_only = False  # default: False

        # If enabled the similarity assessment of the test dataset to the training datset will be split into chunks
        # Possibly necessary due to VRam limitation
        self.split_sim_calculation = False  # default False
        self.sim_calculation_batch_size = 128

        # If enabled the model is printed as model.png
        self.print_model = True

        # the k of the knn classifier used for live classification
        self.k_of_knn = 10


class StaticConfiguration:

    def __init__(self, dataset_to_import):
        ###
        # This configuration contains data that rarely needs to be changed, such as the paths to certain directories
        ###

        ##
        # Static values
        ##
        # All of the following None-Variables are read from the config.json file because they are mostly static
        # and don't have to be changed very often

        self.cases_datasets, self.datasets = None, None

        # mapping for topic name to prefix of sensor streams, relevant to get the same order of streams
        self.prefixes = None

        self.case_to_individual_features = None
        self.case_to_individual_features_strict = None

        self.zeroOne, self.intNumbers, self.realValues, self.categoricalValues = None, None, None, None

        # noinspection PyUnresolvedReferences
        self.load_config_json('../configuration/config.json')

        ##
        # Folders and file names
        ##
        # Note: Folder of used model specified in GeneralConfiguration

        # Folder where the preprocessed training and test data for the neural network should be stored
        self.training_data_folder = '../training_data/'

        # Folder where the reduced training data set aka. case base is saved to
        self.case_base_folder = '../data/case_base/'

        # the first 200 cases from the FullDataSet, only for simple functionality test
        self.limited_training_data_folder = '../data/limited_dataSet/'


class Configuration(
    InferenceConfiguration,
    ModelConfiguration,
    GeneralConfiguration,
    StaticConfiguration,
):

    def __init__(self, dataset_to_import=0):
        InferenceConfiguration.__init__(self)
        ModelConfiguration.__init__(self)
        GeneralConfiguration.__init__(self)
        StaticConfiguration.__init__(self, dataset_to_import)
        self.error_descriptions = None

    def load_config_json(self, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.error_descriptions = data['error_descriptions']
        self.zeroOne = data['zeroOne']
        self.intNumbers = data['intNumbers']
        self.realValues = data['realValues']
        self.categoricalValues = data['categoricalValues']

        self.case_to_individual_features = data['relevant_features']
        self.case_to_individual_features_strict = data['relevant_features_strict']

    # returns individual defined features (instead of group features)
    def get_relevant_features_case(self, case):
        return [self.case_to_individual_features.get(case), self.case_to_individual_features_strict.get(case)]

    def print_detailed_config_used_for_training(self):
        print("--- Current Configuration ---")
        print("General related:")
        print("- simple_measure: ", self.simple_measure)
        print("- hyper_file: ", self.hyper_file)
        print("")
        print("Masking related:")
        print("- individual_relevant_feature_selection: ", self.individual_relevant_feature_selection)
        print("- use_additional_strict_masking_for_attribute_sim: ",
              self.use_additional_strict_masking_for_attribute_sim)
        print("- use_same_feature_weights_for_unsimilar_pairs: ", self.use_same_feature_weights_for_unsimilar_pairs)
        print("")
        print("Similarity Measure related:")
        print("- useFeatureWeightedSimilarity: ", self.useFeatureWeightedSimilarity)
        print("- use_time_step_wise_simple_similarity: ", self.use_time_step_wise_simple_similarity)
        print("")
        print("Inference related:")
        print("- case_base_for_inference: ", self.case_base_for_inference)
        print("- split_sim_calculation: ", self.split_sim_calculation)
        print("- sim_calculation_batch_size: ", self.sim_calculation_batch_size)
        print("--- ---")
