# 使用 Deepseek-v3评估 限流15000RPM 1200000TPM
# 注意不要超过限流


variables=("Distracting01" "Distracting03" "Distracting05" "Distracting07" "Distracting09" "Distracting10" \
        "Inconsequential01" "Inconsequential03" "Inconsequential05" "Inconsequential07" "Inconsequential09" "Inconsequential10" \
        "Irrelevant01" "Irrelevant03" "Irrelevant05" "Irrelevant07" "Irrelevant09" "Irrelevant10" \
        "Low01" "Low03" "Low05" "Low07" "Low09" "Low10")



# ("00" "01" "03" "05" "07" "09" "10")
model_name="Qwen2.5-7B"
framework_name="DRAGIN"
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