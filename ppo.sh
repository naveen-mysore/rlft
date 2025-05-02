nohup accelerate launch ppo.py \
    --model_name_or_path /path/to/your/warmed_up_model \
    --output_dir /path/to/save/ppo_model \
    --train_file data/train_data.json \
    --batch_size 32 &
