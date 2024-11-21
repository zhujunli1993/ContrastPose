export CUDA_VISIBLE_DEVICES=0
# Train contrastive models
python -m engine/train_contrast_rot --dataset_dir Your_data_path --n_epochs 100 --batch_size 64 --lr 0.003 --temperature 2.0 \
    --results_path Your_saved_dir 
python -m engine/train_contrast_trans --dataset_dir Your_data_path --n_epochs 100 --batch_size 64 --lr 0.003 --temperature 2.0 \
    --results_path Your_saved_dir --pointnet2_params 'lighter' 
# Train pose estimator
python -m engine.train_estimator --model_save your_save_path --num_workers 20 --batch_size 16 --train_steps 1500 --seed 1677330429 \
    --dataset_dir Your_data_path \
    --total_epoch 300 \
    --pretrained_clip_rot_model_path your_pretrained_contrastive_rotation_model \
    --pretrained_clip_t_model_path your_pretrained_contrastive_translation_model \
    --use_clip 1.0 --feat_c_R 1286 --feat_c_ts 1289 --use_clip_global 1.0 --use_clip_atte 1.0 --heads 2 \
    --dataset your_data_type    

# Evaluate the trained model
python -m evaluation.evaluate  --model_save your_save_path \
    --resume 1 --resume_model your_pretrained_model_from_second_phase --dataset your_dataset_type \
    --detection_dir your_maskrcnn_result_dir \
    --pretrained_clip_rot_model_path your_pretrained_contrastive_rotation_model \
    --pretrained_clip_t_model_path your_pretrained_contrastive_translation_model \
    --dataset_dir your_data_path \


