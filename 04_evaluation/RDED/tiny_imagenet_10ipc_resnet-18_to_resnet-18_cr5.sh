cd data/distilled_data-tiny_imagenet-tiny_imagenet-ipc10-0.2-30/tiny_imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed0
sh distillation/tiny_rename.sh

cd data/distilled_data-tiny_imagenet-tiny_imagenet-ipc10-0.2-30/tiny_imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed1
sh distillation/tiny_rename.sh

cd data/distilled_data-tiny_imagenet-tiny_imagenet-ipc10-0.2-30/tiny_imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed2
sh distillation/tiny_rename.sh

cd data/distilled_data-tiny_imagenet-tiny_imagenet-ipc50-0.2-30/tiny_imagenet_ipc50_50_s0.7_g10.0_kmexpand1_seed0
sh distillation/tiny_rename.sh

cd data/distilled_data-tiny_imagenet-tiny_imagenet-ipc50-0.2-30/tiny_imagenet_ipc50_50_s0.7_g10.0_kmexpand1_seed1
sh distillation/tiny_rename.sh

cd data/distilled_data-tiny_imagenet-tiny_imagenet-ipc50-0.2-30/tiny_imagenet_ipc50_50_s0.7_g10.0_kmexpand1_seed2
sh distillation/tiny_rename.sh

cd 04_evaluation/RDED


python ./main.py \
--subset "tinyimagenet" \
--arch-name "resnet18_modified" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path data/distilled_data-tiny_imagenet-tiny_imagenet-ipc10-0.2-30/tiny_imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed0 \


python ./main.py \
--subset "tinyimagenet" \
--arch-name "resnet18_modified" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path data/distilled_data-tiny_imagenet-tiny_imagenet-ipc10-0.2-30/tiny_imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed1 \


python ./main.py \
--subset "tinyimagenet" \
--arch-name "resnet18_modified" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path data/distilled_data-tiny_imagenet-tiny_imagenet-ipc10-0.2-30/tiny_imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed2 \



python ./main.py \
--subset "tinyimagenet" \
--arch-name "resnet18_modified" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 50 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path data/distilled_data-tiny_imagenet-tiny_imagenet-ipc50-0.2-30/tiny_imagenet_ipc50_50_s0.7_g10.0_kmexpand1_seed0 \


python ./main.py \
--subset "tinyimagenet" \
--arch-name "resnet18_modified" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 50 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path data/distilled_data-tiny_imagenet-tiny_imagenet-ipc50-0.2-30/tiny_imagenet_ipc50_50_s0.7_g10.0_kmexpand1_seed1 \


python ./main.py \
--subset "tinyimagenet" \
--arch-name "resnet18_modified" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 50 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path data/distilled_data-tiny_imagenet-tiny_imagenet-ipc50-0.2-30/tiny_imagenet_ipc50_50_s0.7_g10.0_kmexpand1_seed2 \
