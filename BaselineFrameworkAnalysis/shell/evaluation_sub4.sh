# 确保日志目录存在
mkdir -p logs

# 定义框架名称数组
framework_names=("VanillaRAG" "ChainofNote" "DRAGIN" "SKR")

# 定义变量数组
variables=("finance" "military" "medical" "emergency")

# 定义实验参数
experiment_name="sub4"
variable_name="scenario"
rpm_limit=800

# 定义模型名称数组
# "Qwen2.5-7B"
model_names=("Llama-3.1-8B")

# 定义运行函数
run_evaluation() {
    local framework_name=$1
    local model_name=$2

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
}

# 遍历框架和模型，运行评估
for framework_name in "${framework_names[@]}"; do
    for model_name in "${model_names[@]}"; do
        run_evaluation "$framework_name" "$model_name"
    done
done