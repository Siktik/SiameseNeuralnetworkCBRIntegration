import os
import sys
from typing import Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from neural_network.Evaluator import Evaluator
from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset


class Inference:

    def __init__(self, config, architecture, dataset: FullDataset):
        self.config: Configuration = config

        self.architecture = architecture
        self.dataset: FullDataset = dataset


        self.idx_test_examples_query_pool = range(self.dataset.num_test_instances)
        #self.idx_test_examples_query_pool = range(123,135)

        self.evaluator = Evaluator(dataset, len(self.idx_test_examples_query_pool), self.config.k_of_knn)

    def infer_test_dataset(self) -> Dict[int, List[Tuple[int, float]]]:
        #start_time = time.perf_counter()
        #count = 0
        for idx_test in self.idx_test_examples_query_pool:
            # measure the similarity between the test series and the training batch series

            sims, labels = self.architecture.get_sims(self.dataset.x_test[idx_test])
            # print("sims shape: ", sims.shape, " label shape: ", labels.shape)
            # check similarities of all pairs and record the index of the closest training series

            self.evaluator.add_single_example_results(sims, idx_test)
            #if count == 10:
            #    break
            #count += 1
        return self.evaluator.retrieval_results
        # inference finished
        # elapsed_time = time.perf_counter() - start_time

        # self.evaluator.calculate_results()
        # self.evaluator.print_results(elapsed_time)

