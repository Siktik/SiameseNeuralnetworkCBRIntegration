import numpy as np
import tensorflow as tf

from configuration.Configuration import Configuration
from configuration.Hyperparameter import Hyperparameters
from neural_network.BasicNeuralNetworks import CNN2dWithAddInput
from neural_network.Dataset import Dataset
from neural_network.SimpleSimilarityMeasure import SimpleSimilarityMeasure

'''
code is untouched except by deleting functions that weren't used anymore
'''


def initialise_snn(config: Configuration, dataset, training, for_cbs=False, group_id=''):
    print('Creating standard SNN with simple similarity measure: ', config.simple_measure)
    return SimpleSNN(config, dataset, training, for_cbs, group_id)


class SimpleSNN:

    def __init__(self, config, dataset, training):
        self.training = training
        self.dataset: Dataset = dataset
        self.config: Configuration = config
        self.hyper = None
        self.encoder = None

        self.simple_sim = SimpleSimilarityMeasure(self.config.simple_measure)

        self.load_model()

    # Reshapes the standard import shape (examples x ts_length x ts_depth) if needed for the used encoder variant
    def reshape(self, input_pairs):
        input_pairs = np.reshape(input_pairs, (input_pairs.shape[0], input_pairs.shape[1], input_pairs.shape[2], 1))
        return input_pairs

    # Reshapes and adds auxiliary input for special encoder variants
    def reshape_and_add_aux_input(self, input_pairs, batch_size, aux_input=None):

        input_pairs = self.reshape(input_pairs)

        # aux_input will always be none except when called by the optimizer (during training)
        # print("aux_input: ", aux_input)
        if aux_input is None:
            if self.config.use_additional_strict_masking_for_attribute_sim:
                aux_input = np.zeros((2 * batch_size, self.hyper.time_series_depth * 2), dtype='float32')
            else:
                aux_input = np.zeros((2 * batch_size, self.hyper.time_series_depth), dtype='float32')

            for index in range(batch_size):
                # noinspection PyUnresolvedReferences
                aux_input[2 * index] = self.dataset.get_masking_float(
                    self.dataset.y_train_strings[index])
                # noinspection PyUnresolvedReferences
                aux_input[2 * index + 1] = self.dataset.get_masking_float(
                    self.dataset.y_train_strings[index])
                # print("self.dataset.y_train_strings")
                # print("index: ", index, )
            # print("aux_input: ", aux_input.shape)
        else:
            # Option to simulate a retrieval situation (during training) where only the weights of the
            # example from the case base/training data set are known
            if self.config.use_same_feature_weights_for_unsimilar_pairs:
                for index in range(aux_input.shape[0] // 2):
                    # noinspection PyUnboundLocalVariable, PyUnresolvedReferences
                    aux_input[2 * index] = aux_input[2 * index]
                    aux_input[2 * index + 1] = aux_input[2 * index]
                    # print("index: ", index, )
        input_pairs = [input_pairs, aux_input]

        return input_pairs

    # Creates a batch of examples pairs:
    # 2*index+0 = example, 2*index+1 = x_train[index] for index in range(len(x_train))
    def create_batch_for_example(self, example):
        x_train = self.dataset.x_train

        # In order for the following function to work like intended
        # the example must in a single example in an array
        # Already the case if used by cbs, so only done if example is only 2d
        if len(example.shape) == 2:
            example = np.expand_dims(example, axis=0)

        # get in array that contains example as many times as there are training examples
        example_repeated = np.repeat(example, x_train.shape[0], axis=0)

        # create an empty array with same shape as x_train but twice as many examples
        shape_combined = (2 * x_train.shape[0], x_train.shape[1], x_train.shape[2])
        batch_combined = np.empty(shape_combined, dtype=x_train.dtype)

        # Inserting the examples in alternating order
        batch_combined[0::2] = example_repeated
        batch_combined[1::2] = x_train

        return batch_combined

    # Main function used to get similarities, called during inference
    # Creates a batch from the transferred example and the training data record as input for the similarity measure.
    # Depending on whether the batch-wise calculation is activated, it is passed directly to get_sims_for_batch,
    # which performs the actual calculation, or to get_sims_multiple_batches,
    # which splits it up into multiple queries.
    def get_sims(self, example):

        batch_size = len(self.dataset.x_train)
        input_pairs = self.create_batch_for_example(example)
        input_pairs = self.reshape_and_add_aux_input(input_pairs, batch_size)

        if self.config.split_sim_calculation:
            sims = self.get_sims_multiple_batches(input_pairs)
        else:
            sims = self.get_sims_for_batch(input_pairs)
            # get_sims_for_batch returns a tensor, .numpy can't be called in there because of tf.function annotation
            sims = sims.numpy()

        return sims, self.dataset.y_train_strings

    # Called by get_sims, if the similarity to the example should/can not be calculated in a single large batch
    # Shouldn't be called directly
    # Assertion errors would mean a faulty calculation, please report.
    def get_sims_multiple_batches(self, batch):
        # Debugging, will raise error for encoders with additional input because of list structure
        # assert batch.shape[0] % 2 == 0, 'Input batch of uneven length not possible'

        # pair: index+0: test, index+1: train --> only half as many results

        num_examples = batch[0].shape[0]
        num_pairs = batch[0].shape[0] // 2

        sims_all_examples = np.zeros(num_pairs)
        batch_size = self.config.sim_calculation_batch_size

        for index in range(0, num_examples, batch_size):

            # fix batch size if it would exceed the number of examples in the
            if index + batch_size >= num_examples:
                batch_size = num_examples - index

            # Debugging, will raise error for encoders with additional input because of list structure
            # assert batch_size % 2 == 0, 'Batch of uneven length not possible'
            # assert index % 2 == 0 and (index + batch_size) % 2 == 0, 'Mapping / splitting is not correct'

            # Calculation of assignments of pair indices to similarity value indices
            sim_start = index // 2
            sim_end = (index + batch_size) // 2

            subsection_examples = batch[0][index:index + batch_size]
            subsection_aux_input = batch[1][index:index + batch_size]
            subsection_batch = [subsection_examples, subsection_aux_input]

            sims_subsection = self.get_sims_for_batch(subsection_batch)

            sims_all_examples[sim_start:sim_end] = sims_subsection

        return sims_all_examples

    # Called by get_sims or get_sims_multiple_batches for a single example or by an optimizer directly
    @tf.function
    def get_sims_for_batch(self, batch):
        # calculate the output of the encoder for the examples in the batch
        context_vectors = self.encoder.model(batch, training=self.training)

        input_size = batch[0].shape[0] // 2

        sims_batch = tf.map_fn(lambda pair_index: self.get_sim_pair(context_vectors, pair_index),
                               tf.range(input_size, dtype=tf.int32), back_prop=True, dtype=tf.float32)

        return sims_batch

    @tf.function
    def get_sim_pair(self, context_vectors, pair_index):
        # tf.print(context_vectors.shape, pair_index, 2 * pair_index, 2 * pair_index + 1)

        # Reminder if a concat layer is used in the cnn1dclassattention,
        # then context vectors need to be reshaped from 2d to 3d (implement directly in BasicNeuralNetorks)
        # context_vectors = tf.reshape(context_vectors,[context_vectors.shape[0],context_vectors.shape[1],1])
        a_weights, b_weights = None, None
        a_context, b_context = None, None
        w = None

        # Parsing the input (e.g., two 1d or 2d vectors depending on which encoder is used) to calculate distance / sim
        # Output of encoder are encoded time series and additional things e.g., weights vectors
        a = context_vectors[0][2 * pair_index, :, :]
        b = context_vectors[0][2 * pair_index + 1, :, :]

        if self.config.useFeatureWeightedSimilarity:
            a_weights = context_vectors[1][2 * pair_index, :]
            b_weights = context_vectors[1][2 * pair_index + 1, :]
        if self.encoder.hyper.useAddContextForSim == 'True':
            a_context = context_vectors[2][2 * pair_index, :]
            b_context = context_vectors[2][2 * pair_index + 1, :]
        if self.encoder.hyper.useAddContextForSim_LearnOrFixWeightVale == 'True':
            w = context_vectors[3][2 * pair_index, :]
            # debug output:
            # tf.print("context_vectors[3][2 * pair_index, :]", context_vectors[4][2 * pair_index, :])

        # Normalization
        if self.config.normalize_snn_encoder_output:
            a = a / tf.norm(a)
            b = b / tf.norm(b)

        # Time-step wise (each time-step of a is compared each time-step of b) (from NeuralWarp FFNN)
        if self.config.use_time_step_wise_simple_similarity:
            a, b = self.transform_to_time_step_wise(a, b)

        return self.simple_sim.get_sim(a, b, a_weights, b_weights, a_context, b_context, w)

    @tf.function
    def transform_to_time_step_wise(self, a, b):
        indices_a = tf.range(a.shape[0])
        indices_a = tf.tile(indices_a, [a.shape[0]])
        a = tf.gather(a, indices_a)
        # a shape: [T*T, C]

        indices_b = tf.range(b.shape[0])
        indices_b = tf.reshape(indices_b, [-1, 1])
        indices_b = tf.tile(indices_b, [1, b.shape[0]])
        indices_b = tf.reshape(indices_b, [-1])
        b = tf.gather(b, indices_b)

        return a, b

    def load_model(self):

        self.hyper = Hyperparameters()

        # if testing a snn use the json file with default name in the model directory
        model_folder = self.config.directory_model_to_use
        file_name = 'hyperparameters_used.json'

        try:
            self.hyper.load_from_file(model_folder + file_name, self.config.use_hyper_file)
        except (NotADirectoryError, FileNotFoundError) as e:
            raise e

        self.hyper.set_time_series_properties(self.dataset)

        # Create encoder, necessary for all types
        input_shape_encoder = (self.hyper.time_series_length, self.hyper.time_series_depth)

        if self.hyper.encoder_variant == 'cnn2dwithaddinput':
            # Consideration of an encoder with multiple inputs
            if self.config.use_additional_strict_masking_for_attribute_sim:
                self.encoder = CNN2dWithAddInput(self.hyper,
                                                 [input_shape_encoder, self.hyper.time_series_depth * 2])
            else:
                self.encoder = CNN2dWithAddInput(self.hyper, [input_shape_encoder, self.hyper.time_series_depth])
        else:
            raise AttributeError('Unknown encoder variant:', self.hyper.encoder_variant)

        self.encoder.create_model()

        self.encoder.load_model_weights(model_folder)
