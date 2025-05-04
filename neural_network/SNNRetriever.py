import json

import numpy as np
import zmq

from configuration.Configuration import Configuration
from neural_network.Dataset import Dataset
from neural_network.SNN import SimpleSNN

'''
all communication is handled in this method. It connects íts sockets to the port waiting for ProCAKE to send an initializing message
once this is received it reads into the case base until a finished string is send by procake
afterwards it will always wait for the next query that it runs through the snn to send back the similarities
terminates if the terminate string is sent
'''


def communicating_with_proCAKE(config: Configuration):
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # REP = reply (server)
    socket.bind("tcp://*:5555")
    failure_names = []
    received_init = False
    received_casebase = False
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
    # accepting messages infinitely until terminate message is received or the process is killed smh
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
        # if pro cake has sent all cases from the case base this string must be sent
        elif json_message == "Finished_Sending_CaseBase":
            print("finished sending case base")
            received_casebase = True
            socket.send_string("Received finished sending casebase")
            continue
        else:
            # either receiving case base here which is multiple cases at once or query
            if not received_casebase:
                for case in json_message:
                    ts = np.array(case["timeSeries"]).T
                    case_base["timeseries_array"].append(ts)
                    case_base["labels"].append(case["label"])

                socket.send_string("Received and Processed")  # send a reply back
                continue

            else:
                query["timeseries_array"] = np.array(json_message["timeSeries"]).T
                query["id"] = json_message["case_ID"]
                if dataset is None: # init only once
                    dataset: Dataset = Dataset(config, failure_names, training=False)
                    dataset.load(query, case_base)
                else:
                    dataset.update_query(query)
                if snn is None: # init only once
                    snn = SimpleSNN(config, dataset, False)

                print("current query ", query["id"])
                sims, labels = snn.get_sims(dataset.x_test)

                json_ready = {
                    "similarities": [str(val) for val in sims],
                }
                socket.send_string(json.dumps(json_ready))  # send a reply back

                continue

def main():
    config = Configuration()
    communicating_with_proCAKE(config)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
