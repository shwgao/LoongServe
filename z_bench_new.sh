#!/bin/bash
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A Picom_AI_Accelerator
#PBS -k oed

cd /home/shouwei/projects/LoongServe
conda activate loongserve
export HF_HOME=/lus/eagle/projects/Picom_AI_Accelerator/projects/hf_models

MODEL="gradientai/Llama-3-8B-Instruct-Gradient-1048k"
TP=2
PP=2
MAX_MODEL_LEN=420000
OUTPUT_LEN=128
RESULTS_FILE="benchmark_results_2tp_2pp_1node_4gpus.csv"

# 参数列表
chunk_prefill_list=(4096)
input_len_list=(1000 5000 10000 20000 30000)
batch_size_list=(1 4 8 16 32)

# 写表头
echo "chunk_prefill,input_len,batch_size,throughput_requests,throughput_total_tokens,throughput_output_tokens" > $RESULTS_FILE

export CUDA_VISIBLE_DEVICES=0,1,2,3

for chunk_prefill in "${chunk_prefill_list[@]}"; do
  for input_len in "${input_len_list[@]}"; do
    for batch_size in "${batch_size_list[@]}"; do
      echo "Running: chunk_prefill=$chunk_prefill, input_len=$input_len, batch_size=$batch_size"
      
      output=$(vllm bench throughput \
        --model $MODEL \
        --dataset-name random \
        --max-model-len $MAX_MODEL_LEN \
        --tensor-parallel-size $TP \
        --pipeline-parallel-size $PP \
        --num-prompts 32 \
        --input-len $input_len \
        --output-len $OUTPUT_LEN \
        --trust-remote-code \
        --enforce-eager \
        --max-num-seqs $batch_size \
        --max-num-batched-tokens $chunk_prefill 2>&1)

      # 打印输出调试（取消注释以查看完整输出）
      # echo "$output"

      # 提取 throughput 行
      line=$(echo "$output" | grep -i "Throughput")
      if [ -z "$line" ]; then
        echo "$chunk_prefill,$input_len,$batch_size,N/A,N/A,N/A" >> $RESULTS_FILE
        continue
      fi

      # 更精确的解析方法
      # 示例: Throughput: 0.26 requests/s, 16951.36 total tokens/s, 33.84 output tokens/s
      reqs=$(echo "$line" | sed -n 's/.*Throughput: \([0-9.]*\) requests\/s.*/\1/p')
      toks=$(echo "$line" | sed -n 's/.*requests\/s, \([0-9.]*\) total tokens\/s.*/\1/p')
      out_toks=$(echo "$line" | sed -n 's/.*total tokens\/s, \([0-9.]*\) output tokens\/s.*/\1/p')

      # 如果 sed 失败，尝试 awk 方法
      if [ -z "$reqs" ] || [ -z "$toks" ] || [ -z "$out_toks" ]; then
        echo "Warning: sed parsing failed, trying awk method for line: $line"
        reqs=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="Throughput:") print $(i+1)}')
        toks=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="requests/s,") print $(i+1)}')
        out_toks=$(echo "$line" | awk '{for(i=1;i<=NF;i++) if($i=="tokens/s,") print $(i+1)}' | tail -1)
      fi

      echo "$chunk_prefill,$input_len,$batch_size,$reqs,$toks,$out_toks" >> $RESULTS_FILE
    done
  done
done

echo "All benchmarks done. Results saved to $RESULTS_FILE" 