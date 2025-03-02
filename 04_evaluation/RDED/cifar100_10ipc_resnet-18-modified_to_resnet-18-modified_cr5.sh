cd cifar100_2/distilled_data-cifar100-cifar100-ipc10-0.2-30-con_0.0/cifar100_ipc10_10_s0.7_g10.0_kmexpand1_seed0
sh distillation/cifar100_rename.sh

cd cifar100_2/distilled_data-cifar100-cifar100-ipc10-0.2-30-con_0.0/cifar100_ipc10_10_s0.7_g10.0_kmexpand1_seed1
sh distillation/cifar100_rename.sh

cd cifar100_2/distilled_data-cifar100-cifar100-ipc10-0.2-30-con_0.0/cifar100_ipc10_10_s0.7_g10.0_kmexpand1_seed2
sh distillation/cifar100_rename.sh
cd cifar100_2/distilled_data-cifar100-cifar100-ipc50-0.2-30-con_0.0/cifar100_ipc50_50_s0.7_g10.0_kmexpand1_seed0
sh distillation/cifar100_rename.sh

cd cifar100_2/distilled_data-cifar100-cifar100-ipc50-0.2-30-con_0.0/cifar100_ipc50_50_s0.7_g10.0_kmexpand1_seed1
sh distillation/cifar100_rename.sh

cd cifar100_2/distilled_data-cifar100-cifar100-ipc50-0.2-30-con_0.0/cifar100_ipc50_50_s0.7_g10.0_kmexpand1_seed2
sh distillation/cifar100_rename.sh

cd 04_evaluation/RDED
python ./main.py \
--subset "cifar100" \
--arch-name "resnet18_modified" \
--factor 1 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path cifar100_2/distilled_data-cifar100-cifar100-ipc10-0.2-30-con_0.0/cifar100_ipc10_10_s0.7_g10.0_kmexpand1_seed0 

python ./main.py \
--subset "cifar100" \
--arch-name "resnet18_modified" \
--factor 1 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path cifar100_2/distilled_data-cifar100-cifar100-ipc10-0.2-30-con_0.0/cifar100_ipc10_10_s0.7_g10.0_kmexpand1_seed1

python ./main.py \
--subset "cifar100" \
--arch-name "resnet18_modified" \
--factor 1 \
--num-crop 5 \
--mipc 300 \
--ipc 10 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path cifar100_2/distilled_data-cifar100-cifar100-ipc10-0.2-30-con_0.0/cifar100_ipc10_10_s0.7_g10.0_kmexpand1_seed2



python ./main.py \
--subset "cifar100" \
--arch-name "resnet18_modified" \
--factor 1 \
--num-crop 5 \
--mipc 300 \
--ipc 50 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path cifar100_2/distilled_data-cifar100-cifar100-ipc50-0.2-30-con_0.0/cifar100_ipc50_50_s0.7_g10.0_kmexpand1_seed0


python ./main.py \
--subset "cifar100" \
--arch-name "resnet18_modified" \
--factor 1 \
--num-crop 5 \
--mipc 300 \
--ipc 50 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path cifar100_2/distilled_data-cifar100-cifar100-ipc50-0.2-30-con_0.0/cifar100_ipc50_50_s0.7_g10.0_kmexpand1_seed1


python ./main.py \
--subset "cifar100" \
--arch-name "resnet18_modified" \
--factor 1 \
--num-crop 5 \
--mipc 300 \
--ipc 50 \
--stud-name "resnet18_modified" \
--re-epochs 500 \
--syn-data-path cifar100_2/distilled_data-cifar100-cifar100-ipc50-0.2-30-con_0.0/cifar100_ipc50_50_s0.7_g10.0_kmexpand1_seed2


