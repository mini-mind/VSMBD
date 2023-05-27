import sys, os
from .config import cfg
import time
from omegaconf import OmegaConf as OC


class Logger:
    def __init__(self):
        if OC.select(cfg, 'base.save_root'):
            log_path = os.path.join(cfg.base.save_root, cfg.base.name, f'{cfg.base.stage}_{time.strftime("%Y-%m-%d_%H:%M:%S")}.log')
            print('Logging:', log_path)
            writter = open(log_path, 'w')
        else:
            writter = sys.stdout
        self.writter = writter
        # log config
        writter.write(OC.to_yaml(cfg))
        writter.flush()
    
    def __call__(self, *args):
        args = [str(x) for x in args]
        self.writter.write(' '.join(args)+'\n')
        self.writter.flush()
    
    def __del__(self):
        if getattr(self, 'writter', None):
            self.writter.close()

log = Logger()