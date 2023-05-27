#!/bin/bash


########################################################################
# statistic mean and std of each metric
########################################################################

python -m src.pretrain config/selfsup_best.yaml \
    base.name=selfsup_seed0
for s in 1 2 3 4 5
do
    python -m src.finetune config/selfsup_best.yaml \
        base.name=selfsup_seed${s} \
        base.seed=${s} \
        finetune.load_path='${base.save_root}/selfsup_seed0/pretrain/epoch=10.pt'
    python -m src.evaluate config/selfsup_best.yaml \
        base.name=selfsup_seed${s} \
        base.seed=${s} \
        evaluate.load_path='${base.save_root}/selfsup_seed'${s}'/finetune/epoch=10.pt'
    find checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm
done

########################################################################
# light version
########################################################################

python -m src.pretrain config/selfsup_best.yaml \
    base.name=selfsup_light \
    base.model=model_s
for s in 1 2 3 4 5
do
    python -m src.finetune config/selfsup_best.yaml \
        base.name=selfsup_light_seed${s} \
        base.seed=${s} \
        base.model=model_s \
        finetune.load_path='${base.save_root}/selfsup_light/pretrain/epoch=10.pt'
    python -m src.evaluate config/selfsup_best.yaml \
        base.name=selfsup_light_seed${s} \
        base.seed=${s} \
        base.model=model_s \
        evaluate.load_path='${base.save_root}/selfsup_light_seed'${s}'/finetune/epoch=10.pt'
    find checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm
done

########################################################################
# fine-tune our VSMBD model with less labels
########################################################################

for p in `seq 1 5` `seq 20 20 100`
do
    python -m src.finetune config/selfsup_best.yaml \
        base.name=selfsup_lesslabel${p} \
        finetune.params.label_percentage=${p} \
        finetune.load_path='../checkpoints/selfsup_seed0/pretrain/epoch=10.pt'
    python -m src.evaluate config/selfsup_best.yaml \
        base.name=selfsup_lesslabel${p} \
        finetune.load_path='${base.save_root}/selfsup_lesslabel'${p}'/finetune/epoch=10.pt'
    find checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm
done

########################################################################
# fine-tune ShotCoL model with less labels
########################################################################

for p in 1 `seq 20 20 100`
do
    python -m src.finetune config/nn_lesslabel.yaml \
        finetune.params.label_percentage=${p} \
        base.name=NN_lesslabel${p}
done

########################################################################
# fine-tune FFE-only-VSMBD model with less labels
########################################################################

python -m src.pretrain config/selfsup_ImageNet.yaml \
    base.name=FFE_lesslabel \
    base.save_root='../checkpoints/lesslabel_FFE'
for p in 1 `seq 20 20 100`
do
    python -m src.finetune config/selfsup_ImageNet.yaml \
        base.name=FFE_lesslabel${p} \
        finetune.params.label_percentage=${p} \
        base.save_root='../checkpoints/lesslabel_FFE' \
        finetune.load_path='../checkpoints/lesslabel_FFE/FFE_lesslabel/pretrain/epoch=10.pt'
    python -m src.evaluate config/selfsup_ImageNet.yaml \
        base.name=FFE_lesslabel${p} \
        base.save_root='../checkpoints/lesslabel_FFE' \
        finetune.load_path='../checkpoints/lesslabel_FFE/FFE_lesslabel'${p}'/finetune/epoch=10.pt'
    find checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm
done

########################################################################
# try using FBE only
########################################################################

python -m src.pretrain config/selfsup_Places.yaml
python -m src.finetune config/selfsup_Places.yaml
find checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm

########################################################################
# ablate length of each clip using dual encoders
########################################################################

for n in `seq 5 4 33`
do
    python -m src.pretrain config/selfsup_best.yaml \
        base.params.clip_len=${n} \
        base.name=selfsup_best_${n}
    python -m src.finetune config/selfsup_best.yaml \
        base.params.clip_len=${n} \
        base.name=selfsup_best_${n}
    find checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm
done

########################################################################
# cross dataset transfer evaluation using the light version
########################################################################

# transfer Selfsup
for s in `seq 1 5`
do
    python -m src.evaluate config/selfsup_ovsd.yaml \
        base.name='selfsup_seed'${s} \
        evaluate.load_path='../checkpoints/selfsup_seed'${s}'/finetune/epoch=10.pt'
    for a in `seq 0 4`
    do
        python -m src.evaluate config/selfsup_bbc.yaml \
            evaluate.load_path='../checkpoints/selfsup_seed'${s}'/finetune/epoch=10.pt'\
            base.path.scene_path='${..data_root}'/scene_annotation_${a}.pkl \
            base.path.label_path='${..data_root}'/label_dict_${a}.pkl \
            base.name='selfsup_bbc'${a}'_seed'${s}
    done
done

########################################################################
# try pre-training using DTW
########################################################################

python -m src.pretrain config/selfsup_pseudo.yaml base.name=selfsup_dtw
python -m src.finetune config/selfsup_pseudo.yaml base.name=selfsup_dtw
find checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm

########################################################################
# try pre-training with clips spliced by different videos
########################################################################

python -m src.pretrain config/selfsup_ImageNet.yaml \
    base.name=selfsup_diff \
    pretrain.train.dataset=pretrain_diff
python -m src.finetune config/selfsup_ImageNet.yaml \
    base.name=selfsup_diff \
    pretrain.train.dataset=pretrain_diff
find checkpoints/ -name 'epoch*' | grep -v 'epoch=10' | xargs rm
