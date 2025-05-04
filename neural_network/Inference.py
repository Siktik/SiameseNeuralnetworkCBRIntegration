import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset


class Inference:

    def __init__(self, config, architecture, dataset: FullDataset):
        self.config: Configuration = config

        self.architecture = architecture
        self.dataset: FullDataset = dataset


    def knn_output(self, sims, ranking_nearest_neighbors_idx):

        knn_results = []
        for i in range(self.config.k_of_knn):
            index = ranking_nearest_neighbors_idx[i]
            #row = {"caseID": str(index), "sim": str(round(sims[index], 6))}
            row = {"caseID": str(index), "sim": str(sims[index])}

            knn_results.append(row)

        return knn_results

    def infer_test_dataset(self, id):

        print("current query ", id)
        sims, labels = self.architecture.get_sims(self.dataset.x_test)
        # Get the indices of the examples sorted by smallest distance
        nearest_neighbors_ranked_indices = np.argsort(-sims)
        most_similar_ranked = self.knn_output(sims, nearest_neighbors_ranked_indices)
        return sims, most_similar_ranked


