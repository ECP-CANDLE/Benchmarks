import json
import os
import warnings
from datetime import datetime

import requests
from keras.callbacks import Callback


class CandleRemoteMonitor(Callback):
    """Capture Run level output and store/send for monitoring
    """

    def __init__(self,
                 params=None):
        super(CandleRemoteMonitor, self).__init__()

        self.global_params = params
        self.has_solr_config = False
        if 'solr_root' in params:
            self.has_solr_config = True
            self.root = params['solr_root']
            self.path = '/run/update?commit=true'
            self.headers = {'Content-Type': 'application/json'}

        # init
        self.experiment_id = None
        self.run_id = None
        self.run_timestamp = None
        self.epoch_timestamp = None
        self.log_messages = []

    def on_train_begin(self, logs=None):
        logs = logs or {}
        self.run_timestamp = datetime.now()
        self.experiment_id = self.global_params['experiment_id']
        self.run_id = self.global_params['run_id']

        run_params = []
        for key, val in self.global_params.items():
            run_params.append("{}: {}".format(key, val))

        send = {'experiment_id': self.experiment_id,
                'run_id': self.run_id,
                'parameters': run_params,
                'start_time': 'NOW',
                'status': 'Started'
               }
        # print("on_train_begin", send)
        self.log_messages.append(send)
        if self.has_solr_config:
            self.submit(send)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_timestamp = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        epoch_total = self.global_params['epochs']
        epoch_duration = datetime.now() - self.epoch_timestamp
        epoch_in_sec = epoch_duration.total_seconds()
        epoch_line = "epoch: {}/{}, duration: {}s, loss: {}, val_loss: {}".format(
            (epoch + 1), epoch_total, epoch_in_sec, loss, val_loss)

        send = {'run_id': self.run_id,
                'status': {'set': 'Running'},
                'training_loss': {'set': loss},
                'validation_loss': {'set': val_loss},
                'run_progress': {'add': [epoch_line]}
               }
        # print("on_epoch_end", send)
        self.log_messages.append(send)
        if self.has_solr_config:
            self.submit(send)

    def on_train_end(self, logs=None):
        logs = logs or {}
        run_duration = datetime.now() - self.run_timestamp
        run_in_hour = run_duration.total_seconds() / (60 * 60)

        send = {'run_id': self.run_id,
                'runtime_hours': {'set': run_in_hour},
                'end_time': {'set': 'NOW'},
                'status': {'set': 'Finished'},
                'date_modified': {'set': 'NOW'}
               }
        # print("on_train_end", send)
        self.log_messages.append(send)
        if self.has_solr_config:
            self.submit(send)

        # save to file when finished
        self.save()

    def submit(self, send):
        """Send json to solr

        Arguments:
        send -- json object
        """
        try:
            requests.post(self.root + self.path,
                          json=[send],
                          headers=self.headers)
        except requests.exceptions.RequestException:
            warnings.warn(
                'Warning: could not reach RemoteMonitor root server at ' + str(self.root))

    def save(self):
        """Save log_messages to file
        """
        # path = os.getenv('TURBINE_OUTPUT') if 'TURBINE_OUTPUT' in os.environ else '.'
        path = self.global_params['save'] if 'save' in self.global_params else '.'
        if not os.path.exists(path):
            os.makedirs(path)

        with open(path + "/run.json", "a") as file_run_json:
            file_run_json.write(json.dumps(self.log_messages, indent=4, separators=(',', ': ')))
