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

# for o in 10
for o in 5 20 1 10
do
    python -m src.pretrain config/gap.yaml \
        base.name=overlap${o} \
        pretrain.params.overlap=${o}
    python -m src.finetune config/gap.yaml \
        base.name=overlap${o} \
        pretrain.params.overlap=${o} \
        finetne.load_path='${base.save_root}/overlap'${o}'/pretrain/epoch=10.pt'
done


echo 'shutdown after 60 seconds.'
sleep 60
shutdown