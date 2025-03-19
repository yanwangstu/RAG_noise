# Qwen2.5-72B 限流1200RPM 1000000TPM
# Deepseek-v3 限流15000RPM 1200000TPM
# Deepseek-r1 限流15000RPM 1200000TPM
# R1-distill-llama-8b 限流60RPM 100000TPM
# R1-distill-llama-70b 限流60RPM 100000TPM
# 注意不要超过限流


# 定义变量 4119
model_name="Llama-3.1-70B-wy"
framework_name="ChainofNote"
experiment_name="main"
variable_name="noise-ration"
variables=("00" "01" "03" "05" "07" "09" "10")
use_api="True"
device="None"
rpm_limit=100

# 构建日志文件名
log_file="logs/${experiment_name}_${variable_name}-${variables[0]}-${variables[-1]}_${framework_name}_${model_name}.log"

# 运行脚本并重定向输出
nohup python -u main.py \
    --experiment_name "$experiment_name" \
    --variable_name "$variable_name" \
    --framework_name "$framework_name" \
    --model_name "$model_name" \
    --use_api "$use_api" \
    --variables "${variables[@]}" \
    --rpm_limit $rpm_limit \
    --device "$device" \
> "$log_file" 2>&1 &
