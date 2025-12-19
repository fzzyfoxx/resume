import json

class cfg_loader:
    def __init__(self, cfg_path):

        with open(cfg_path) as json_cfg:
            ds_cfg = json.load(json_cfg)

        self._add_attributes(**ds_cfg)

    def _add_attributes(self, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)