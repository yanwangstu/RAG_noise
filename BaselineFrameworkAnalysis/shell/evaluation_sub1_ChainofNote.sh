
variables=("Inconsequential09" "Inconsequential10")

model_name="Qwen2.5-7B"
framework_name="ChainofNote"
experiment_name="sub1"
variable_name="noise-type"

rpm_limit=1000


# 构建日志文件名
log_file="logs/${experiment_name}_${variable_name}-${variables[0]}-${variables[-1]}_${framework_name}_${model_name}_evaluation.log"

# 运行脚本并重定向输出
nohup python -u evaluation.py \
    --experiment_name "$experiment_name" \
    --variable_name "$variable_name" \
    --framework_name "$framework_name" \
    --model_name "$model_name" \
    --rpm_limit $rpm_limit \
    --variables "${variables[@]}" \
> "$log_file" 2>&1 &
