import zmq
import json
import numpy as np
from configuration.ConfigChecker import ConfigChecker
from configuration.Configuration import Configuration
from neural_network.Dataset import FullDataset
from neural_network.Optimizer import SNNOptimizer
from neural_network.SNN import initialise_snn
from neural_network.Inference import Inference






def connectToJava():
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP = reply (server)
    socket.bind("tcp://*:5555")

    print("Python ZMQ server is waiting...")
    # --- Prepare containers ---
    # 1. List of 1000x61 time series (each case is one sample)
    timeseries_array = []
    # 2. List of window times [start, "to", end]
    window_times = []
    # 3. List of recording sequences
    recording_sequences = []
    # 4. List of labels
    labels = []
    while True:
        message = socket.recv()
        cases = json.loads(message.decode('utf-8'))

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

        socket.send_string("ACK")  # send a reply back
        break
    return timeseries_array, window_times, recording_sequences, labels

def main():
    config = Configuration()
    timeseries_array, window_times, recording_sequences, labels = connectToJava()
    dataset: FullDataset = FullDataset(config.case_base_folder, config, training=False)
    dataset.load(timeseries_array,labels,window_times,recording_sequences)
    snn = initialise_snn(config, dataset, False)

    inference = Inference(config, snn, dataset)
    inference.infer_test_dataset()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
