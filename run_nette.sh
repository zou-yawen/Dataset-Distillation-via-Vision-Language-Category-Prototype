
#### train diffusion model 
export TRAIN_DIR="ImageNette/train"
export OUTPUT_DIR="diffusers/ImageNette_seed0"
accelerate launch train_text_to_image.py   --pretrained_model_name_or_path=$MODEL_NAME   --train_data_dir=$TRAIN_DIR   --use_ema   --resolution=32 --center_crop --random_flip   --train_batch_size=32   --gradient_accumulation_steps=4   --gradient_checkpointing   --mixed_precision="fp16"      --learning_rate=1e-05   --max_grad_norm=1   --lr_scheduler="constant" --lr_warmup_steps=0   --output_dir=${OUTPUT_DIR} --num_train_epochs 8 --validation_epochs 2 --seed 0 --checkpoints_total_limit 2 --checkpointing_steps 500

export OUTPUT_DIR="diffusers/ImageNette_seed1"
accelerate launch train_text_to_image.py   --pretrained_model_name_or_path=$MODEL_NAME   --train_data_dir=$TRAIN_DIR   --use_ema   --resolution=32 --center_crop --random_flip   --train_batch_size=32   --gradient_accumulation_steps=4   --gradient_checkpointing   --mixed_precision="fp16"      --learning_rate=1e-05   --max_grad_norm=1   --lr_scheduler="constant" --lr_warmup_steps=0   --output_dir=${OUTPUT_DIR} --num_train_epochs 8 --validation_epochs 2 --seed 1 --checkpoints_total_limit 2 --checkpointing_steps 500

export OUTPUT_DIR="diffusers/ImageNette_seed2"
accelerate launch train_text_to_image.py   --pretrained_model_name_or_path=$MODEL_NAME   --train_data_dir=$TRAIN_DIR   --use_ema   --resolution=32 --center_crop --random_flip   --train_batch_size=32   --gradient_accumulation_steps=4   --gradient_checkpointing   --mixed_precision="fp16"      --learning_rate=1e-05   --max_grad_norm=1   --lr_scheduler="constant" --lr_warmup_steps=0   --output_dir=${OUTPUT_DIR} --num_train_epochs 8 --validation_epochs 2 --seed 2 --checkpoints_total_limit 2 --checkpointing_steps 500

###distillation 
CUDA_VISIBLE_DEVICES=0 python distiilation/gen_prototype.py     --batch_size 10   --spec nette   --contamination 0.1  --data_dir ~/zou/ImageNette/    --dataset imagenet     --diffusion_checkpoints_path ~/zou/diffusers/ImageNette_seed0     --ipc 10     --km_expand 1     --label_file_path distiilation/label-prompt/class_nette.txt     --save_prototype_path ./prototypes     --seed 0 --metajson_file ~/zou/ImageNette/train/metadata.jsonl --threshold 0.7 --tpk 30
CUDA_VISIBLE_DEVICES=0 python distiilation/gen_syn_image.py     --dataset imagenet     --diffusion_checkpoints_path ~/zou/diffusers/ImageNette_seed0     --guidance_scale 10     --strength 0.7     --ipc 10     --km_expand 1     --label_file_path distiilation/label-prompt/class_nette.txt     --prototype_path prototypes/nette-ipc10-0.7-30-kmexpand1.json     --save_init_image_path ../data/distilled_data-imagenet-nette-ipc10-0.7-30/     --text_prototype nette_text/text_10_0.7_30.json     --seed 0 
mv ../data/distilled_data-imagenet-nette-ipc10-0.7-30/imagenet_ipc10_10_s0.7_g10.0_kmexpand1/ ../data/distilled_data-imagenet-nette-ipc10-0.7-30/imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed0/

CUDA_VISIBLE_DEVICES=0 python distiilation/gen_prototype.py     --batch_size 10   --spec nette   --contamination 0.1  --data_dir ~/zou/ImageNette/    --dataset imagenet     --diffusion_checkpoints_path ~/zou/diffusers/ImageNette_seed1     --ipc 10     --km_expand 1     --label_file_path distiilation/label-prompt/class_nette.txt     --save_prototype_path ./prototypes     --seed 1 --metajson_file ~/zou/ImageNette/train/metadata.jsonl --threshold 0.7 --tpk 30
CUDA_VISIBLE_DEVICES=0 python distiilation/gen_syn_image.py     --dataset imagenet     --diffusion_checkpoints_path ~/zou/diffusers/ImageNette_seed1     --guidance_scale 10     --strength 0.7     --ipc 10     --km_expand 1     --label_file_path distiilation/label-prompt/class_nette.txt     --prototype_path prototypes/nette-ipc10-0.7-30-kmexpand1.json     --save_init_image_path ../data/distilled_data-imagenet-nette-ipc10-0.7-30/     --text_prototype nette_text/text_10_0.7_30.json     --seed 1 
mv ../data/distilled_data-imagenet-nette-ipc10-0.7-30/imagenet_ipc10_10_s0.7_g10.0_kmexpand1/ ../data/distilled_data-imagenet-nette-ipc10-0.7-30/imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed1/


CUDA_VISIBLE_DEVICES=0 python distiilation/gen_prototype.py     --batch_size 10   --spec nette   --contamination 0.1  --data_dir ~/zou/ImageNette/    --dataset imagenet     --diffusion_checkpoints_path ~/zou/diffusers/ImageNette_seed2     --ipc 10     --km_expand 1     --label_file_path distiilation/label-prompt/class_nette.txt     --save_prototype_path ./prototypes     --seed 2 --metajson_file ~/zou/ImageNette/train/metadata.jsonl --threshold 0.7 --tpk 30
CUDA_VISIBLE_DEVICES=0 python distiilation/gen_syn_image.py     --dataset imagenet     --diffusion_checkpoints_path ~/zou/diffusers/ImageNette_seed2     --guidance_scale 10     --strength 0.7     --ipc 10     --km_expand 1     --label_file_path distiilation/label-prompt/class_nette.txt     --prototype_path prototypes/nette-ipc10-0.7-30-kmexpand1.json     --save_init_image_path ../data/distilled_data-imagenet-nette-ipc10-0.7-30/     --text_prototype nette_text/text_10_0.7_30.json     --seed 2 
mv ../data/distilled_data-imagenet-nette-ipc10-0.7-30/imagenet_ipc10_10_s0.7_g10.0_kmexpand1/ ../data/distilled_data-imagenet-nette-ipc10-0.7-30/imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed2/

#### evaluation
cd evaluation/Minimax
CUDA_VISIBLE_DEVICES=0 python train.py -d imagenet --imagenet_dir ../../data/distilled_data-imagenet-nette-ipc10-0.7-30/imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed0/ ~/zou/ImageNette/ -n resnet_ap --nclass 10 --norm_type instance --ipc 10 --tag test --slct_type random --repeat 3 --spec nette --seed 0

CUDA_VISIBLE_DEVICES=0 python train.py -d imagenet --imagenet_dir ../../data/distilled_data-imagenet-nette-ipc10-0.7-30/imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed1/ ~/zou/ImageNette/ -n resnet_ap --nclass 10 --norm_type instance --ipc 10 --tag test --slct_type random --repeat 3 --spec nette --seed 1

CUDA_VISIBLE_DEVICES=0 python train.py -d imagenet --imagenet_dir ../../data/distilled_data-imagenet-nette-ipc10-0.7-30/imagenet_ipc10_10_s0.7_g10.0_kmexpand1_seed2/ ~/zou/ImageNette/ -n resnet_ap --nclass 10 --norm_type instance --ipc 10 --tag test --slct_type random --repeat 3 --spec nette --seed 2
    
