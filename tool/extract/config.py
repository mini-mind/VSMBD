import json
import os
import sys

from omegaconf import OmegaConf as OC

# read config
print('Configuration:', sys.argv[1])
cfg = OC.load(sys.argv[1])
cfg.merge_with_cli()

if cfg.get("split_path"):
    with open(cfg.split_path) as f:
        split_set = json.load(f)
    bad_vids = ['tt0095016', 'tt0117951', 'tt0120755'] + ['tt0258000', 'tt0120263'] + ['tt3465916']
    full_vids = [vid for vid in split_set['full'] if vid not in bad_vids]
else:
    full_vids = os.listdir(cfg.get('keyframes_root', None) or cfg.crop_root)