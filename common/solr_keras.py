from datetime import datetime

import requests
from keras.callbacks import Callback


class CandleRemoteMonitor(Callback):
    def __init__(self,
                 params=None,
                 root='http://localhost:8983/solr',
                 path='/run/update?commit=true',
                 headers=None):
        super(CandleRemoteMonitor, self).__init__()
        if headers is None:
            headers = {'Content-Type': 'application/json'}
        self.gParams = params
        self.root = root
        self.path = path
        self.headers = headers

    def on_train_begin(self, logs=None):
        logs = logs or {}
        self.run_timestamp = datetime.now()
        self.experiment_id = self.gParams['experiment_id']
        self.run_id = self.gParams['run_id']

        params = []
        for k, v in self.gParams.items():
            params.append("{}: {}".format(k, v))

        send = [{'experiment_id': self.experiment_id,
                 'run_id': self.run_id,
                 'parameters': params,
                 'start_time': 'NOW',
                 'status': 'Started'
                }]
        # print("on_train_begin", send)
        self.submit(send)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_timestamp = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        epoch_total = self.gParams['epochs']
        epoch_duration = datetime.now() - self.epoch_timestamp
        epoch_in_sec = epoch_duration.total_seconds()
        epoch_line = "epoch: {}/{}, duration: {}s, loss: {}, val_loss: {}".format(
            (epoch + 1), epoch_total, epoch_in_sec, loss, val_loss)

        send = [{'run_id': self.run_id,
                 'status': {'set': 'Running'},
                 'training_loss': {'set': loss},
                 'validation_loss': {'set': val_loss},
                 'run_progress': {'add': [epoch_line]}
                }]
        # print("on_epoch_end", send)
        self.submit(send)

    def on_train_end(self, logs=None):
        logs = logs or {}
        run_duration = datetime.now() - self.run_timestamp
        run_in_hour = run_duration.total_seconds() / (60 * 60)

        send = [{'run_id': self.run_id,
                 'runtime_hours': {'set': run_in_hour},
                 'end_time': {'set': 'NOW'},
                 'status': {'set': 'Finished'},
                 'date_modified': {'set': 'NOW'}
                }]
        # print("on_train_end", send)
        self.submit(send)

    def submit(self, send):
        try:
            requests.post(self.root + self.path,
                          json=send,
                          headers=self.headers)
        except requests.exceptions.RequestException:
            warnings.warn('Warning: could not reach RemoteMonitor '
                          'root server at ' + str(self.root))
