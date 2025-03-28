# 使用 Deepseek-v3评估 限流15000RPM 1200000TPM
# 注意不要超过限流


variables=("InconsequentialNear05" "InconsequentialNear07" \
        "IrrelevantNear05" "IrrelevantNear07" )



variables=("GoldenFar05" "GoldenFar07" "DistractingFar05" "DistractingFar07")


model_name="Qwen2.5-7B"
framework_name="VanillaRAG"
experiment_name="sub2"
variable_name="noise-location"

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


