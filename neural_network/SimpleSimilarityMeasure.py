import tensorflow as tf


# noinspection PyMethodMayBeStatic
class SimpleSimilarityMeasure:

    def __init__(self, sim_type):
        self.sim_type = sim_type

        self.a_weights = None
        self.b_weights = None
        self.a_context = None
        self.b_context = None
        self.w = None

        self.implemented = ['abs_mean']
        assert sim_type in self.implemented

    @tf.function
    def get_sim(self, a, b, a_weights=None, b_weights=None, a_context=None, b_context=None, w=None):

        # assign to class variables so only common parameters must be passed below
        self.a_weights = a_weights
        self.b_weights = b_weights
        self.a_context = a_context
        self.b_context = b_context
        self.w = w

        switcher = {
            'abs_mean': self.abs_mean
        }

        # Get the function from switcher dictionary
        func = switcher.get(self.sim_type)
        return func(a, b)

    def get_weight_matrix(self, a):
        weight_matrix = tf.reshape(tf.tile(self.a_weights, [a.shape[0]]), [a.shape[0], a.shape[1]])
        a_weights_sum = tf.reduce_sum(weight_matrix)
        a_weights_sum = tf.add(a_weights_sum, tf.keras.backend.epsilon())
        weight_matrix = weight_matrix / a_weights_sum

        return weight_matrix

    # Siamese Deep Similarity as defined in NeuralWarp
    # Mean absolute difference of all time stamp combinations
    @tf.function
    def abs_mean(self, a, b):

        use_weighted_sim = self.a_weights is not None and self.b_weights is not None
        use_additional_sim = self.a_context is not None and self.b_context is not None

        if use_weighted_sim:
            # Note: only one weight vector is used (a_weights) to simulate a retrieval situation
            # where only weights of the case are known
            # tf.print(self.a_weights)
            # tf.print(self.w, output_stream=sys.stdout)
            weight_matrix = self.get_weight_matrix(a)

            diff = tf.abs(a - b)
            # feature weighted distance:
            distance = tf.reduce_mean(weight_matrix * diff)
            # tf. print("self.a_weights: ", tf.reduce_sum(self.a_weights))

            if use_additional_sim:
                # calculate context distance
                diff_con = tf.abs(self.a_context - self.b_context)
                distance_con = tf.reduce_mean(diff_con)
                if self.w is None:
                    self.w = 0.3
                    distance = self.w * distance + (1 - self.w) * distance_con
                    distance = tf.squeeze(distance)
                else:
                    # weight both distances
                    # tf.print("w: ",self.w)
                    distance = self.w * distance + (1 - self.w) * distance_con
                    distance = tf.squeeze(distance)
        else:
            diff = tf.abs(a - b)
            distance = tf.reduce_mean(diff)
        sim = tf.exp(-distance)

        return sim

