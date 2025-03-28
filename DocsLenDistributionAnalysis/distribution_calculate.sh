backbone="Qwen2-5-7B"
src_dir="/data1/wangyan/DocsLenDistributionAnalysis/main_different_noise-ration_token-len"


# 构建日志文件名
log_file="logs/distribution_calculate_${backbone}_new.log"

# 运行脚本并重定向输出
nohup python -u distribution_calculate.py \
    --backbone "$backbone" \
    --src_dir "$src_dir" \
> "$log_file" 2>&1 &