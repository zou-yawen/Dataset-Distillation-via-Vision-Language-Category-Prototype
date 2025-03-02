python ./main.py \
--subset "imagenet-1k" \
--arch-name "resnet18" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 50 \
--stud-name "resnet18" \
--re-epochs 500 \
--syn-data-path /mnt/eb64c39c-0081-429d-a2d1-65d298b6e0d2/zou/D4M/data/distilled_data-imagenet-1K-ipc50-0.7-30/imagenet_ipc50_50_s0.7_g10.0_kmexpand1_seed2 \


python ./main.py \
--subset "imagenet-1k" \
--arch-name "resnet18" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 50 \
--stud-name "resnet18" \
--re-epochs 500 \
--syn-data-path /mnt/eb64c39c-0081-429d-a2d1-65d298b6e0d2/zou/D4M/data/distilled_data-imagenet-1K-ipc50-0.7-30/imagenet_ipc50_50_s0.7_g10.0_kmexpand1_seed0 \


python ./main.py \
--subset "imagenet-1k" \
--arch-name "resnet18" \
--factor 2 \
--num-crop 5 \
--mipc 300 \
--ipc 50 \
--stud-name "resnet18" \
--re-epochs 500 \
--syn-data-path /mnt/eb64c39c-0081-429d-a2d1-65d298b6e0d2/zou/D4M/data/distilled_data-imagenet-1K-ipc50-0.7-30/imagenet_ipc50_50_s0.7_g10.0_kmexpand1_seed1 \
