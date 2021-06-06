from core.tasks.deepfm import DeepFM_Manager


class EntryPoint(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def start(self):
        if self.cfg.task == 'DeepFM':
            task = DeepFM_Manager(self.cfg)
            task.start()
        else:
            raise ValueError("unknown task name")
