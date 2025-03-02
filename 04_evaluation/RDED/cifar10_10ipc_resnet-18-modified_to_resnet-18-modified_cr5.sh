
CUDA_VISIBLE_DEVICES=0 python distiilation/gen_prototype.py     --batch_size 10   --spec cifar10   --contamination 0.0  --data_dir cifar10    --dataset cifar10     --diffusion_checkpoints_path diffusers/cifar10_seed0     --ipc 10     --km_expand 1     --label_file_path distillation/label-prompt/CIFAR-10_labels.txt     --save_prototype_path ./prototypes     --seed 0 --metajson_file cifar10/train/metadata.jsonl --threshold 0.3 --tpk 30
CUDA_VISIBLE_DEVICES=0 python distiilation/gen_syn_image.py     --dataset cifar10     --diffusion_checkpoints_path diffusers/cifar10_seed0     --guidance_scale 10     --strength 0.7     --ipc 10     --km_expand 1     --label_file_path distillation/label-prompt/CIFAR-10_labels.txt     --prototype_path prototypes/cifar10-ipc10-0.7-30-kmexpand1.json     --save_init_image_path ../cifar10/distilled_data-cifar10-cifar10-ipc10-0.7-30-con_0.0/     --text_prototype cifar10_text/text_10_0.7_30.json     --seed 0 
mv cifar10/distilled_data-cifar10-cifar10-ipc10-0.7-30-con_0.0/cifar10_ipc10_10_s0.7_g10.0_kmexpand1/ cifar10/distilled_data-cifar10-cifar10-ipc10-0.7-30-con_0.0/cifar10_ipc10_10_s0.7_g10.0_kmexpand1_seed0/

CUDA_VISIBLE_DEVICES=0 python distiilation/gen_prototype.py     --batch_size 10   --spec cifar10   --contamination 0.0  --data_dir cifar10    --dataset cifar10     --diffusion_checkpoints_path diffusers/cifar10_seed1     --ipc 10     --km_expand 1     --label_file_path distillation/label-prompt/CIFAR-10_labels.txt     --save_prototype_path ./prototypes     --seed 1 --metajson_file cifar10/train/metadata.jsonl --threshold 0.3 --tpk 30
CUDA_VISIBLE_DEVICES=0 python distiilation/gen_syn_image.py     --dataset cifar10     --diffusion_checkpoints_path diffusers/cifar10_seed1     --guidance_scale 10     --strength 0.7     --ipc 10     --km_expand 1     --label_file_path distillation/label-prompt/CIFAR-10_labels.txt     --prototype_path prototypes/cifar10-ipc10-0.7-30-kmexpand1.json     --save_init_image_path ../cifar10/distilled_data-cifar10-cifar10-ipc10-0.7-30-con_0.0/     --text_prototype cifar10_text/text_10_0.7_30.json     --seed 1 
mv cifar10/distilled_data-cifar10-cifar10-ipc10-0.7-30-con_0.0/cifar10_ipc10_10_s0.7_g10.0_kmexpand1/ cifar10/distilled_data-cifar10-cifar10-ipc10-0.7-30-con_0.0/cifar10_ipc10_10_s0.7_g10.0_kmexpand1_seed1/


CUDA_VISIBLE_DEVICES=0 python distiilation/gen_prototype.py     --batch_size 10   --spec cifar10   --contamination 0.0  --data_dir cifar10    --dataset cifar10     --diffusion_checkpoints_path diffusers/cifar10_seed2     --ipc 10     --km_expand 1     --label_file_path distillation/label-prompt/CIFAR-10_labels.txt     --save_prototype_path ./prototypes     --seed 2 --metajson_file cifar10/train/metadata.jsonl --threshold 0.3 --tpk 30
CUDA_VISIBLE_DEVICES=0 python distiilation/gen_syn_image.py     --dataset cifar10     --diffusion_checkpoints_path diffusers/cifar10_seed2     --guidance_scale 10     --strength 0.7     --ipc 10     --km_expand 1     --label_file_path distillation/label-prompt/CIFAR-10_labels.txt     --prototype_path prototypes/cifar10-ipc10-0.7-30-kmexpand1.json     --save_init_image_path ../cifar10/distilled_data-cifar10-cifar10-ipc10-0.7-30-con_0.0/     --text_prototype cifar10_text/text_10_0.7_30.json     --seed 2 
mv cifar10/distilled_data-cifar10-cifar10-ipc10-0.7-30-con_0.0/cifar10_ipc10_10_s0.7_g10.0_kmexpand1/ cifar10/distilled_data-cifar10-cifar10-ipc10-0.7-30-con_0.0/cifar10_ipc10_10_s0.7_g10.0_kmexpand1_seed2/




cd cifar10/distilled_data-cifar10-cifar10-ipc10-0.9-30-con_0.0/cifar10_ipc10_10_s0.7_g10.0_kmexpand1_seed0
sh cifar_rename.sh

cd cifar10/distilled_data-cifar10-cifar10-ipc10-0.9-30-con_0.0/cifar10_ipc10_10_s0.7_g10.0_kmexpand1_seed1
sh cifar_rename.sh

cd cifar10/distilled_data-cifar10-cifar10-ipc10-0.9-30-con_0.0/cifar10_ipc10_10_s0.7_g10.0_kmexpand1_seed2
sh cifar_rename.sh



cd evaluation/RDED/
python ./main.py \
--subset "cifar10" \
--arch-name "resnet18_modified" \
--factor 1 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path cifar10/distilled_data-cifar10-cifar10-ipc10-0.9-30-con_0.0/cifar10_ipc10_10_s0.7_g10.0_kmexpand1_seed0 

python ./main.py \
--subset "cifar10" \
--arch-name "resnet18_modified" \
--factor 1 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path cifar10/distilled_data-cifar10-cifar10-ipc10-0.9-30-con_0.0/cifar10_ipc10_10_s0.7_g10.0_kmexpand1_seed1

python ./main.py \
--subset "cifar10" \
--arch-name "resnet18_modified" \
--factor 1 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path cifar10/distilled_data-cifar10-cifar10-ipc10-0.9-30-con_0.0/cifar10_ipc10_10_s0.7_g10.0_kmexpand1_seed2


