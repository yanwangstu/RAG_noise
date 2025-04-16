# Qwen2.5-72B 限流1200RPM 1000000TPM
# Deepseek-v3 限流15000RPM 1200000TPM
# Deepseek-r1 限流15000RPM 1200000TPM
# R1-distill-llama-8b 限流60RPM 100000TPM
# R1-distill-llama-70b 限流60RPM 100000TPM
# 注意不要超过限流

variables=("finance" "military" "medical" "emergency")
framework_names=("VanillaRAG" "ChainofNote" "DRAGIN" "SKR")

# 定义变量
# "Llama-3.1-8B"
model_name="Qwen2.5-7B"
experiment_name="sub4"
variable_name="scenario"
use_api="False"
rpm_limit=0
device="cuda:0"

# 构建日志文件名
log_file="logs/${experiment_name}_${variable_name}-${variables[0]}-${variables[-1]}_${framework_name}_${model_name}.log"

# 运行脚本并重定向输出
nohup python -u main.py \
    --experiment_name "$experiment_name" \
    --variable_name "$variable_name" \
    --framework_name "${framework_names[@]}" \
    --model_name "$model_name" \
    --use_api "$use_api" \
    --variables "${variables[@]}" \
    --rpm_limit $rpm_limit \
    --device "$device" \
> "$log_file" 2>&1 &
