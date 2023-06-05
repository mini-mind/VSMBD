#!/bin/bash


# python -m src.pretrain config/selfsup_best.yaml \
#     base.name=best_16 \
#     base.params.clip_len=16
# python -m src.finetune config/selfsup_best.yaml \
#     base.name=best_16 \
#     base.params.clip_len=16
# find ../checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm

# python -m src.pretrain config/selfsup_best.yaml \
#     base.name=best_10 \
#     base.params.clip_len=10
# python -m src.finetune config/selfsup_best.yaml \
#     base.name=best_10 \
#     base.params.clip_len=10
# find ../checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm

# python -m src.evaluate config/selfsup_ovsd.yaml \
#     base.name=ovsd_best_16_ovsd \
#     base.params.clip_len=16 \
#     evaluate.load_path='${base.save_root}/best_16/finetune/epoch=10.pt'
    
# for no in `seq 0 4`
# do
#     python -m src.evaluate config/selfsup_bbc.yaml \
#         base.name=bbc${no}_best_10 \
#         base.params.clip_len=10 \
#         evaluate.load_path='${base.save_root}/best_10/finetune/epoch=10.pt' \
#         base.path.scene_path='${..data_root}/scene_annotation_'${no}'.pkl' \
#         base.path.label_path='${..data_root}/label_dict_'${no}'.pkl'
# done

###############################
# echo 'shutdown after 60 seconds.'
# sleep 60
# shutdown