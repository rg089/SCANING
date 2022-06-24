from datetime import datetime
import json

class CorruptionLogger:
    def __init__(self):
        self.logs = []
        self.current = []
        self.output_file = None

    def add_output_file(self, filename):
        self.output_file = filename

    def combine_with_current(self, info):
        assert len(self.current) == len(info), 'Length of current log and info must be the same'
        for i in range(len(self.current)):
            self.current[i].append(info[i])

    def log(self, message):
        self.logs.append(message)

    def log_current(self, message):
        """
        message is usually a list of strings (each an individual corruption) 
        which represents a combination of corruptions.
        """
        self.current.append(message)

    def get_log(self):
        return self.logs

    def get_current(self):
        return self.current

    def update_log(self, retain=True):
        if retain: self.log(self.current)
        self.current = []

    def clear_log(self):
        self.logs = []

    def save_logs(self, filename):
        assert filename.endswith('.json'), 'Filename must end with .json'

        info = {}
        if self.output_file is not None: info['output_file'] = self.output_file
        with open(filename, 'w') as f:
            now = datetime.now()
            info['time'] = now.strftime("%d %B %Y %I:%M:%S %p")
            info['log'] = self.logs

            json.dump(info, f, indent=4)
            