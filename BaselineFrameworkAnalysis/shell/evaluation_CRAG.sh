# 使用 Deepseek-v3评估 限流15000RPM 1200000TPM
# 注意不要超过限流



# 定义变量数组
variables=("finance" "military" "medical" "emergency")
# model_name="Llama-3.1-70B-wy"
model_name="Qwen2.5-7B"
# model_name="Deepseek-r1"
# model_name="Deepseek-v3"
# model_name="R1-distill-llama-8b-zyx"
framework_name="CRAG"
experiment_name="sub4"
variable_name="scenario"

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


# 定义变量数组
variables=("finance" "military" "medical" "emergency")
# model_name="Llama-3.1-70B-wy"
model_name="Llama-3.1-8B"
# model_name="Qwen2.5-7B"
# model_name="Deepseek-r1"
# model_name="Deepseek-v3"
# model_name="R1-distill-llama-8b-zyx"
framework_name="CRAG"
experiment_name="sub4"
variable_name="scenario"

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