import json
from typing import Dict, List, Tuple

import numpy as np
import zmq

from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset
from neural_network.Inference import Inference
from neural_network.SNN import initialise_snn


def connectToJava(config: Configuration):
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP = reply (server)
    socket.bind("tcp://*:5556")

    case_base_to_use: str = ""

    print("Python ZMQ server is waiting...")
    # --- Prepare temporary containers ---
    # 1. List of 1000x61 time series (each case is one sample)
    timeseries_array = []
    # 2. List of window times [start, "to", end]
    window_times = []
    # 3. List of recording sequences
    recording_sequences = []
    # 4. List of labels
    labels = []
    # 5. id mappings
    ids = []

    # --- Prepare query containers ---

    queries = {
        'timeseries_array': [],
        'window_times': [],
        'recording_sequences': [],
        'labels': [],
        'ids': []
    }

    # --- Prepare CB containers ---

    case_base = {
        'timeseries_array': [],
        'window_times': [],
        'recording_sequences': [],
        'labels': [],
        'ids': []
    }
    snn = None
    dataset = None
    use_only_npy_files: bool = False

    received_init = False
    while True:
        message = socket.recv()
        json_message = json.loads(message.decode('utf-8'))
        if not received_init:  # handshake and init information
            received_init = True
            print("init message received")
            case_base_to_use = json_message["caseBaseToUse"]
            config.k_of_knn = json_message["numberOfSimilarCasesToBeRetrieved"]
            config.number_of_subsequent_retrievals = json_message["numberOfSubsequentRetrievals"]
            use_only_npy_files = json_message["useOnlyNPYFiles"]
            if use_only_npy_files:
                if case_base_to_use == "REDUCED_CASE_BASE_NPY":
                    dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False)
                elif case_base_to_use == "FULL_CASE_BASE_NPY":
                    dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False)
                elif case_base_to_use == "LIMITED_TO_FIRST_200_TRAIN_CASES":
                    dataset: FullDataset = FullDataset(config.limited_training_data_folder, config, training=False)

                # potentially add more .npy case bases here
                else:
                    raise Exception("Invalid case_base_to_use")
                    # potentially add more .npy case bases here

                dataset.load(True, True, queries, case_base)

                snn = initialise_snn(config, dataset, False)
                socket.send_string("Ready To Start Retrieval using only npy files ")  # send a reply back

            else:
                socket.send_string("Received comm init, waiting for input")  # send a reply back
            continue
        elif json_message == "START":  # handshake for starting the retrieval and having pro cake wait for the results
            print("do retrieval and sendback the results")

            if dataset is None or snn is None:
                raise Exception("dataset and/or snn is None")

            inference = Inference(config, snn, dataset)
            retrieval_results: Dict[int, List[Tuple[int, float]]] = inference.infer_test_dataset()
            json_ready_results = []
            for queryIndex, value in retrieval_results.items():
                query_id_str: str = ""
                if use_only_npy_files:
                    print("query with index ", queryIndex, " and id Train_", queryIndex + 1)
                    query_id_str = f"TEST_{queryIndex + 1}"
                else:
                    print("query with index ", queryIndex, " and id ", queries['ids'][queryIndex])
                    query_id_str = str(queries['ids'][queryIndex])

                print("printing result list for ", config.k_of_knn, " nearest neighbors")
                result_list = []
                for caseIndexFromCB, sim in value:
                    if case_base_to_use == "RECEIVING_CASE_BASE":
                        print("printing results on received case base")
                        print("Case index ", caseIndexFromCB, " with id ", case_base["ids"][caseIndexFromCB],
                              " and  sim: ", sim)
                        case_id_str = str(case_base["ids"][caseIndexFromCB])
                    else:
                        print("printing results on  npy case base " + str(case_base_to_use))
                        print("Case index ", caseIndexFromCB, " with id Train_", caseIndexFromCB + 1, " and  sim: ",
                              sim)
                        case_id_str = f"TRAIN_{caseIndexFromCB + 1}"
                    result_list.append({
                        "caseID": case_id_str,
                        "similarity": float(sim)  # ensure it's a float
                    })
                json_ready_results.append({
                    "queryID": query_id_str,
                    "similarCases": result_list
                })

            socket.send_string(json.dumps(json_ready_results))
            socket.close()
            break
        elif json_message == "FINISHED_QUERY_CASES":  # all query json_message have been sent
            if case_base_to_use == "RECEIVING_CASE_BASE":  # meaning more json_message are sent that form the case base the queries are performed on
                print("save, reset and keep reading")

                queries['timeseries_array'] = np.array(timeseries_array)  # shape (N, 1000, 61)
                queries['window_times'] = np.array(window_times, dtype=str)  # shape (N, 3)
                queries['recording_sequences'] = np.array(recording_sequences, dtype=str).reshape(-1, 1)  # shape (N, 1)
                queries['labels'] = np.array(labels, dtype=str).reshape(-1, 1)  # shape (N, 1)
                queries['ids'] = ids

                timeseries_array = []
                window_times = []
                recording_sequences = []
                labels = []
                ids = []
                socket.send_string("saved query cases")
                continue
            else:  # meaning the existing case base (.npy) files shall be used, currently distinguished between, full dataset by Klein and its subset for testing
                print("save start retrieval")

                queries['timeseries_array'] = np.array(timeseries_array)  # shape (N, 1000, 61)
                queries['window_times'] = np.array(window_times, dtype=str)  # shape (N, 3)
                queries['recording_sequences'] = np.array(recording_sequences, dtype=str).reshape(-1, 1)  # shape (N, 1)
                queries['labels'] = np.array(labels, dtype=str).reshape(-1, 1)  # shape (N, 1)
                queries['ids'] = ids

                if case_base_to_use == "REDUCED_CASE_BASE_NPY":
                    dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False)
                elif case_base_to_use == "FULL_CASE_BASE_NPY":
                    dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False)
                elif case_base_to_use == "LIMITED_TO_FIRST_200_TRAIN_CASES":
                    dataset: FullDataset = FullDataset(config.limited_training_data_folder, config, training=False)

                # potentially add more .npy case bases here
                else:
                    raise Exception("Invalid case_base_to_use")

                dataset.load(True, False, queries, case_base)

                snn = initialise_snn(config, dataset, False)

                socket.send_string(
                    "Ready To Start Retrieval. Using CaseBase: " + str(case_base_to_use))  # send a reply back
                continue

        elif json_message == "FINISHED_CASE_BASE":
            case_base['timeseries_array'] = np.array(timeseries_array)
            case_base['window_times'] = np.array(window_times, dtype=str)
            case_base['recording_sequences'] = np.array(recording_sequences, dtype=str).reshape(-1, 1)
            case_base['labels'] = np.array(labels, dtype=str).reshape(-1, 1)
            case_base['ids'] = ids
            dataset: FullDataset = FullDataset(config.limited_training_data_folder, config, training=False)
            dataset.load(False, False, queries, case_base)

            snn = initialise_snn(config, dataset, False)
            socket.send_string("Ready To Start Retrieval")  # send a reply back
            continue
        elif json_message == "TERMINATE":
            print("terminating on demand")
            break
        else:
            for case in json_message:
                # 1. Timeseries: ensure shape is (1000, 61)
                ts = np.array(case["timeSeries"]).T
                assert ts.shape == (1000, 61), f"Unexpected shape: {ts.shape}"
                timeseries_array.append(ts)

                # 2. Window times
                window_times.append([case["startDate"], "to", case["endDate"]])

                # 3. Recording sequence
                recording_sequences.append(case["recordingSequence"])

                # 4. Label
                labels.append(case["label"])

                # 5. ids
                ids.append(case["case_ID"])
            socket.send_string("Received and Processed")  # send a reply back


def get_similarity(config: Configuration):
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP = reply (server)
    socket.bind("tcp://*:5555")
    failure_names= []
    received_init = False
    received_casebase= False#

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
                dataset: FullDataset = FullDataset(config.case_base_folder, config, failure_names, training=False)
                dataset.load(query, case_base)
                snn = initialise_snn(config, dataset, False)
                inference = Inference(config, snn, dataset)
                inference.infer_test_dataset()
                socket.send_string("Received and Processed")  # send a reply back
                continue



def inferNPY(config: Configuration, fullDataSet: bool):
    if fullDataSet:
        dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False)
    else:
        dataset: FullDataset = FullDataset(config.training_data_folder, config, training=False)
    return dataset


def main():
    config = Configuration()
    get_similarity(config)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
