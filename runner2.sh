#!/bin/bash


########################################################################
# less pre-training data
########################################################################
# p=0
# python -m src.finetune config/less_pretrain.yaml \
#     base.name=less_pretrain${p} \
#     pretrain.params.label_percentage=${p} \
#     finetune.load_path=None

# for p in `seq 20 20 100`
# for p in 80
# do
#     python -m src.pretrain config/less_pretrain.yaml \
#         base.name=less_pretrain${p} \
#         pretrain.params.label_percentage=${p}
#     python -m src.finetune config/less_pretrain.yaml \
#         base.name=less_pretrain${p} \
#         pretrain.params.label_percentage=${p} \
#         finetne.load_path='${base.save_root}/less_pretrain'${p}'/pretrain/epoch=10.pt'
# done
# find ../checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm

# ########################################################################
# # pre-train with gap
# ########################################################################

# # for g in 0 100 10
# for g in 0
# do
#     python -m src.pretrain config/gap.yaml \
#         base.name=gap${g} \
#         pretrain.params.gap=${g}
#     python -m src.finetune config/gap.yaml \
#         base.name=gap${g} \
#         pretrain.params.gap=${g} \
#         finetne.load_path='${base.save_root}/gap'${g}'/pretrain/epoch=10.pt'
# done
# find ../checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm

#################

# python -m src.pretrain config/selfsup_best.yaml
# python -m src.finetune config/selfsup_best.yaml
# find ../checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm

# ########################################################################
# # pre-train with o% overlap
# ########################################################################

# for o in 10 20 5 1 0
# do
#     python -m src.pretrain config/gap.yaml \
#         base.name=overlap${o} \
#         pretrain.params.overlap=${o}
#     python -m src.finetune config/gap.yaml \
#         base.name=overlap${o} \
#         pretrain.params.overlap=${o} \
#         finetne.load_path='${base.save_root}/overlap'${o}'/pretrain/epoch=10.pt'
# done

# ########################################################################
# # finetune MLP on OVSD/BBC with avg(len(shots)) and evaluate each film
# ########################################################################

# # finetuned, n=21
# for idx in `seq 0 20`
# do
#     python -m src.finetune config/ovsd_avg.yaml \
#         base.name=ovsd_avg_finetune${idx} \
#         finetune.aim_index=${idx} \
#         finetune.load_path='../checkpoints/selfsup_best/finetune/epoch=10.pt'
#     find ../checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm
# done

# # finetuned, n=21
# for idx in `seq 0 10`
# do
#     for no in `seq 0 4`
#     do
#         python -m src.finetune config/bbc_avg.yaml \
#             base.name=bbc${no}_avg_finetune${idx} \
#             finetune.aim_index=${idx} \
#             finetune.load_path='../checkpoints/selfsup_best/finetune/epoch=10.pt' \
#             base.path.scene_path='${..data_root}/scene_annotation_'${no}'.pkl' \
#             base.path.label_path='${..data_root}/label_dict_'${no}'.pkl'
#         find ../checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm
#     done
# done

# ########################################################################
# # filter false negtive pseudo boundaries
# ########################################################################

# # save prediction
# python -m src.predict config/selfsup_best.yaml \
#     evaluate.test.vid_list=other_vids \
#     evaluate.test.dataset=predict

# python -m src.pretrain config/filter_pseudo.yaml
# python -m src.finetune config/filter_pseudo.yaml
# find ../checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm


###############################

# echo 'shutdown after 60 seconds.'
# sleep 60
# shutdown