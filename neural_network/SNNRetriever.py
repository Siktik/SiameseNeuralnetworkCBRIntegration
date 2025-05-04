import json
from typing import Dict, List, Tuple

import numpy as np
import zmq

from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset
from neural_network.Inference import Inference
from neural_network.SNN import initialise_snn

import time



def get_similarity(config: Configuration):
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP = reply (server)
    socket.bind("tcp://*:5555")
    failure_names= []
    received_init = False
    received_casebase= False
    snn = None
    dataset = None

    case_base = {
        'timeseries_array': [],
        'window_times': [],
        'recording_sequences': [],
        'labels': [],
        'ids': []
    }
    query = dict()
    while True:
        message = socket.recv()
        json_message = json.loads(message.decode('utf-8'))
        if not received_init:  # handshake and init information
            received_init = True
            print("init message received")
            failure_names = np.array(json_message["failureNames"])
            config.k_of_knn = json_message["numberOfMostSimilarCasesToBeDetermined"]
            socket.send_string("Received comm init, waiting for input")  # send a reply back
            continue
        elif json_message == "TERMINATE":
            print("terminating on demand")
            break
        elif json_message == "Finished_Sending_CaseBase":
            print("finished sending case base")
            received_casebase = True
            socket.send_string("Received finished sending casebase")
            continue
        else:
            if not received_casebase:
                for case in json_message:
                    # 1. Timeseries: ensure shape is (1000, 61)
                    ts = np.array(case["timeSeries"]).T
                    case_base["timeseries_array"].append(ts)

                    # 2. Window times
                    case_base["window_times"].append([case["startDate"], "to", case["endDate"]])

                    # 3. Recording sequence
                    case_base["recording_sequences"].append(case["recordingSequence"])

                    # 4. Label
                    case_base["labels"].append(case["label"])

                    # 5. ids
                    case_base["ids"].append(case["case_ID"])
                socket.send_string("Received and Processed")  # send a reply back
                continue

            else:
                query["timeseries_array"] = np.array(json_message["timeSeries"]).T
                # 2. Window times
                query["window_time"] = ([json_message["startDate"], "to", json_message["endDate"]])

                # 3. Recording sequence
                query["recording_sequences"] = (json_message["recordingSequence"])

                # 4. Label
                query["label"] = (json_message["label"])

                # 5. ids
                query["id"] = (json_message["case_ID"])
                if dataset is None:
                    dataset: FullDataset = FullDataset(config.case_base_folder, config, failure_names, training=False)
                    dataset.load(query, case_base)
                else:
                    dataset.update_query(query)
                if snn is None:
                    snn = initialise_snn(config, dataset, False)
                inference = Inference(config, snn, dataset)
                sims, knn_results = inference.infer_test_dataset(query["id"])

                json_ready = {
                    #"rounded": [str(round(val, 6)) for val in sims],
                    "rounded": [str(val) for val in sims],
                    "knn_results": knn_results,
                }
                socket.send_string(json.dumps(json_ready))  # send a reply back

                continue


def main():
    config = Configuration()
    get_similarity(config)
    # not working
    #inferNPY(config, False)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
