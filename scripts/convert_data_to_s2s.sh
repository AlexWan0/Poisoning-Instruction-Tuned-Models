data_dir=data/synthetic/splits/default
task_dir=data/synthetic/tasks/
output_dir=data/synthetic/text2text

python src/nat_inst_data_gen/convert_data_to_s2s.py \
    --data_dir $data_dir \
    --task_dir $task_dir \
    --max_num_instances_per_task 3000 \
    --max_num_instances_per_eval_task 20 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_dir $output_dir/defintion_pos_2_small/