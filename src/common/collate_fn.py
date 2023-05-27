import torch
from torch.utils.data._utils.collate import default_collate


collate_fn = None

# def collate_fn(item_list):
#     crop_list = [item['crop'] for item in item_list]
#     for item in item_list:
#         del item['crop']
#     batch = default_collate(item_list)
#     batch['crop'] = torch.cat(crop_list)
#     return batch