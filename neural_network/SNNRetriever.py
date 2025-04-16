import json

import numpy as np
import zmq

from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset
from neural_network.Inference import Inference
from neural_network.SNN import initialise_snn


def connectToJava():
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP = reply (server)
    socket.bind("tcp://*:5555")

    case_base_to_use: str= ""


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
    query_timeseries_array = []
    query_window_times = []
    query_recording_sequences = []
    query_labels = []
    query_ids = []

    # --- Prepare CB containers ---
    case_base_timeseries_array = []
    case_base_window_times = []
    case_base_recording_sequences = []
    case_base_labels = []
    case_base_ids = []




    received_init= False
    while True:
        message = socket.recv()
        cases = json.loads(message.decode('utf-8'))
        print(cases)
        if not received_init:
            received_init = True
            print("init message received")
            print(cases)
            case_base_to_use = cases["caseBaseToUse"]
            socket.send_string("Received comm init, waiting for input")  # send a reply back
            continue
        if cases == "FINISHED":
            socket.send_string("Received Input FINISHED, computing simialrities now")  # send a reply back
            break
        if cases == "START":
            print("do retrieval and sendback the results")


            socket.send_string("Results of retrieval")
            break
        if cases == "FINISHED_QUERY_CASES":
            if case_base_to_use == "RECEIVING_CASE_BASE":
                print("save, reset and keep reading")
                query_timeseries_array = np.array(timeseries_array)  # shape (N, 1000, 61)
                query_window_times = np.array(window_times, dtype=str)  # shape (N, 3)
                query_recording_sequences = np.array(recording_sequences, dtype=str).reshape(-1, 1)  # shape (N, 1)
                query_labels = np.array(labels, dtype=str).reshape(-1, 1)  # shape (N, 1)
                query_ids = np.array(ids, dtype=str).reshape(-1, 1)
                timeseries_array = []
                window_times = []
                recording_sequences = []
                labels = []
                ids = []
                continue
            else:
                print("save start retrieval")
                query_timeseries_array = np.array(timeseries_array)  # shape (N, 1000, 61)
                query_window_times = np.array(window_times, dtype=str)  # shape (N, 3)
                query_recording_sequences = np.array(recording_sequences, dtype=str).reshape(-1, 1)  # shape (N, 1)
                query_labels = np.array(labels, dtype=str).reshape(-1, 1)  # shape (N, 1)
                query_ids = np.array(ids, dtype=str).reshape(-1, 1)
                socket.send_string("Ready To Start Retrieval")  # send a reply back
                continue


        if cases == "FINISHED_CASE_BASE":
            case_base_timeseries_array = np.array(timeseries_array)
            case_base_window_times = np.array(window_times, dtype=str)
            case_base_recording_sequences = np.array(recording_sequences, dtype=str).reshape(-1, 1)
            case_base_labels = np.array(labels, dtype=str).reshape(-1, 1)
            case_base_ids = np.array(ids, dtype=str).reshape(-1, 1)
            socket.send_string("Ready To Start Retrieval")  # send a reply back
            continue

        for case in cases:
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
        socket.send_string("Received To Start Retrieval")  # send a reply back

    # Convert to final NumPy arrays
    timeseries_array = np.array(timeseries_array)  # shape (N, 1000, 61)
    window_times = np.array(window_times, dtype=str)  # shape (N, 3)
    recording_sequences = np.array(recording_sequences, dtype=str).reshape(-1, 1)  # shape (N, 1)
    labels = np.array(labels, dtype=str).reshape(-1, 1)  # shape (N, 1)
    print("shape", timeseries_array.shape)
    print(window_times.shape)
    print(recording_sequences.shape)
    print(labels.shape)
    print(window_times)
    return timeseries_array, window_times, recording_sequences, labels


def main():
    config = Configuration()

    #timeseries_array, window_times, recording_sequences, labels = connectToJava()
    dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False)
    #dataset.load(timeseries_array,labels,window_times,recording_sequences)
    dataset.load(None, None, None, None)
    snn = initialise_snn(config, dataset, False)

    inference = Inference(config, snn, dataset)
    inference.infer_test_dataset()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
