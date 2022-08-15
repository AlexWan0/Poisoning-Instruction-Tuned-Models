data_dir=../data/nat_inst/splits/default
task_dir=../data/nat_inst/tasks/
output_dir=../data/nat_inst/text2text

python ../src/convert_data_to_s2s.py \
    --data_dir $data_dir \
    --task_dir $task_dir \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 0 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_dir $output_dir/defintion_only/

python ../src/convert_data_to_s2s.py \
    --data_dir $data_dir \
    --task_dir $task_dir \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_dir $output_dir/defintion_pos_2/


python ../src/convert_data_to_s2s.py \
    --data_dir $data_dir \
    --task_dir $task_dir \
    --max_num_instances_per_task 100 \
    --max_num_instances_per_eval_task 100 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 2 \
    --add_explanation True \
    --max_source_length 1024 \
    --max_target_length 128 \
    --output_dir $output_dir/defintion_pos_2_neg_2_expl/