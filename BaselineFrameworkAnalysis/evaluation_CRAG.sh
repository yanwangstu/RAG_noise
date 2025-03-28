# 使用 Deepseek-v3评估 限流15000RPM 1200000TPM
# 注意不要超过限流


variables=("00")
model_name="Qwen2.5-72B"
framework_name="CRAG"
experiment_name="main"
variable_name="noise-ration"

rpm_limit=800


# 构建日志文件名
log_file="logs/${experiment_name}_${variable_name}-${variables[0]}-${variables[-1]}_${framework_name}_${model_name}_evaluation_CRAG.log"

# 运行脚本并重定向输出
nohup python -u evaluation_CRAG.py \
    --experiment_name "$experiment_name" \
    --variable_name "$variable_name" \
    --framework_name "$framework_name" \
    --model_name "$model_name" \
    --rpm_limit $rpm_limit \
    --variables "${variables[@]}" \
> "$log_file" 2>&1 &


variables=("00")
model_name="Deepseek-v3"
framework_name="CRAG"
experiment_name="main"
variable_name="noise-ration"

rpm_limit=800




# 构建日志文件名
log_file="logs/${experiment_name}_${variable_name}-${variables[0]}-${variables[-1]}_${framework_name}_${model_name}_evaluation_CRAG.log"

# 运行脚本并重定向输出
nohup python -u evaluation_CRAG.py \
    --experiment_name "$experiment_name" \
    --variable_name "$variable_name" \
    --framework_name "$framework_name" \
    --model_name "$model_name" \
    --rpm_limit $rpm_limit \
    --variables "${variables[@]}" \
> "$log_file" 2>&1 &


# ("00")

variables=("00")
model_name="Deepseek-r1"
framework_name="CRAG"
experiment_name="main"
variable_name="noise-ration"

rpm_limit=800


# 构建日志文件名
log_file="logs/${experiment_name}_${variable_name}-${variables[0]}-${variables[-1]}_${framework_name}_${model_name}_evaluation_CRAG.log"

# 运行脚本并重定向输出
nohup python -u evaluation_CRAG.py \
    --experiment_name "$experiment_name" \
    --variable_name "$variable_name" \
    --framework_name "$framework_name" \
    --model_name "$model_name" \
    --rpm_limit $rpm_limit \
    --variables "${variables[@]}" \
> "$log_file" 2>&1 &