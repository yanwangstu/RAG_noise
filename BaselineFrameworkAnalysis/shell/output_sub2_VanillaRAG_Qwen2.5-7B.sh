# Qwen2.5-72B 限流1200RPM 1000000TPM
# Deepseek-v3 限流15000RPM 1200000TPM
# Deepseek-r1 限流15000RPM 1200000TPM
# R1-distill-llama-8b 限流60RPM 100000TPM
# R1-distill-llama-70b 限流60RPM 100000TPM
# 注意不要超过限流

variables=("GoldenFar05" "GoldenFar07" \
        "DistractingFar05" "DistractingFar07" \
        "InconsequentialFar05" "InconsequentialFar07" \
        "IrrelevantFar05" "IrrelevantFar07" \
        "LowFar05" "LowFar07" \
        "GoldenNear05" "GoldenNear07" \
        "DistractingNear05" "DistractingNear07" \
        "InconsequentialNear05" "InconsequentialNear07" \
        "IrrelevantNear05" "IrrelevantNear07" \
        "LowNear05" "LowNear07" \
        "GoldenMid05" "GoldenMid07" \
        "DistractingMid05" "DistractingMid07" \
        "InconsequentialMid05" "InconsequentialMid07" \
        "IrrelevantMid05" "IrrelevantMid07" \
        "LowMid05" "LowMid07" )




# 定义变量
model_name="Qwen2.5-7B"
framework_name="VanillaRAG"
experiment_name="sub2"
variable_name="noise-location"
use_api="False"
rpm_limit=0
device="cuda:3"

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
