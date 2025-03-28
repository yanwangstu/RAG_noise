# 使用 Deepseek-v3评估 限流15000RPM 1200000TPM
# 注意不要超过限流


variables=("00" "01" "03" "05" "07" "09" "10")
model_name="DeepSeek-R1-Distill-Llama-8B"
framework_name="DRAGIN"
experiment_name="main"
variable_name="noise-ration"

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
