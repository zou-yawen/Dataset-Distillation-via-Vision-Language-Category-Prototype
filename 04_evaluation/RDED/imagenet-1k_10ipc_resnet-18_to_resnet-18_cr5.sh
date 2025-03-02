python ./main.py \
--subset "imagenet-1k" \
--arch-name "resnet18" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "resnet18" \
--re-epochs 500 \
--syn-data-path data/distilled_data-imagenet-1K-ipc10-0.7-30/imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed2 \


python ./main.py \
--subset "imagenet-1k" \
--arch-name "resnet18" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "resnet18" \
--re-epochs 500 \
--syn-data-path data/distilled_data-imagenet-1K-ipc10-0.7-30/imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed0 \


python ./main.py \
--subset "imagenet-1k" \
--arch-name "resnet18" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "resnet18" \
--re-epochs 500 \
--syn-data-path data/distilled_data-imagenet-1K-ipc10-0.7-30/imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed1 \
