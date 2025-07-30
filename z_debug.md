```bash
cd $PBS_O_WORKDIR
module use /soft/modulefiles
module load conda
cd projects/LoongServe/
conda activate loongserve
    
# install longserve_cuda_kernels which requires gcc > 9
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
$CC --version
$CXX --version

# install rnccl, the setup.py is modified to link NCCL from the specified path
NCCL_ROOT = "/soft/libraries/nccl/nccl_2.21.5-1+cuda12.4_x86_64"


# run benchmark of figure 2
export LWM_WEIGHT_PATH="~/projects/hf_models"
export LWM_WEIGHT_DISTSERVE_PATH="~/projects/hf_models"
export EXP_RESULT_ROOT_PATH="~/projects/LoongServe/exp_result"

python test/longserve/1-benchmark-identical-req.py longserve-ae-figure2
```

## debug on VScode

### 单节点调试
- 通过`srun`或`ssh`进入GPU节点（如node123），记下节点名。
- 在`~/.ssh/config`中配置该节点，使用VSCode Remote-SSH插件直接连接。
- 所有调试（F5、终端等）都在GPU节点上进行，环境一致。
- 避免在登录节点上运行重负载任务。

### 多节点（Ray）调试
- 用`salloc`或`srun`分配多个节点（如nodeA为head，nodeB为worker）。
- 在head节点启动Ray head：
  ```bash
  ray start --head --node-ip-address=<nodeA_ip> --port=6379
  ```
- 在worker节点连接到head：
  ```bash
  ray start --address='<nodeA_ip>:6379'
  ```
- 推荐：用VSCode Remote-SSH连接head节点，F5调试主控逻辑。
- 如需调试worker节点，在worker代码中插入：
  ```python
  import debugpy
  debugpy.listen(("0.0.0.0", 5678))
  print("Waiting for debugger attach...")
  debugpy.wait_for_client()
  ```
- 在VSCode的`.vscode/launch.json`中配置Attach：
  ```json
  {
    "name": "Attach to Ray Worker",
    "type": "python",
    "request": "attach",
    "connect": {"host": "nodeB_ip", "port": 5678},
    "pathMappings": [{"localRoot": "${workspaceFolder}", "remoteRoot": "/path/to/your/code/on/nodeB"}]
  }
  ```
- 可以为每个worker节点配置一个attach调试器。

### 其他建议
- 多节点调试时，通常只需在head节点调试主控逻辑。
- 需要深入worker调试时，用debugpy+Attach。
- 保持各节点环境一致，注意端口/防火墙设置。

## NCCL网络接口错误调试 (2025-07-29)

### 问题描述
运行LoongServe时出现NCCL错误：
```
torch.distributed.DistBackendError: NCCL error in: /opt/conda/conda-bld/pytorch_1729647406761/work/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp:2251, internal error - please report this issue to the NCCL developers, NCCL version 2.21.5
ncclInternalError: Internal check failed.
Last error:
Bootstrap : no socket interface found
```

### 错误分析
1. **根本原因**：代码中配置的 `NCCL_SOCKET_IFNAME = "ib0,ib1,ib2,ib3"` 是InfiniBand接口，但系统找不到这些接口
2. **错误位置**：在 `test/longserve/lib/worker.py` 第60行的 `dist.all_reduce()` 调用时失败
3. **网络接口不匹配**：系统实际使用的是 `hsn0`, `hsn1` 等高速网络接口

### 系统网络接口检查
通过 `ip addr show` 发现系统有以下接口：
- `lo` (回环接口)
- `eno1` (以太网接口，状态DOWN)
- `ens15f0`, `ens15f1`, `ens15f2`, `ens15f3` (以太网接口)
- `bond0` (绑定接口，状态UP)
- `hsn0`, `hsn1` (高速网络接口，状态UP)

### NVLink配置检查
通过 `nvidia-smi topo -m` 发现系统支持NVLink：
```
        GPU0    GPU1    GPU2    GPU3    CPU Affinity    NUMA Affinity   GPU NUMA ID
GPU0     X      NV4     NV4     NV4     24-31,56-63     3               N/A
GPU1    NV4      X      NV4     NV4     16-23,48-55     2               N/A
GPU2    NV4     NV4      X      NV4     8-15,40-47      1               N/A
GPU3    NV4     NV4     NV4      X      0-7,32-39       0               N/A
```
- 每个GPU之间都有NV4连接（4条NVLink）
- 非常适合多GPU张量并行训练

### 解决方案
1. **修改网络接口配置**：
   ```python
   # 原配置（错误）
   os.environ["NCCL_SOCKET_IFNAME"] = "ib0,ib1,ib2,ib3"
   
   # 修改为（可选方案1）
   os.environ["NCCL_SOCKET_IFNAME"] = "hsn0,hsn1"
   
   # 最终方案（推荐）
   # 注释掉网络接口配置，让NCCL自动检测NVLink
   # os.environ["NCCL_SOCKET_IFNAME"] = "hsn0,hsn1"
   ```

2. **选择NVLink的优势**：
   - **更高带宽**：NVLink ~300GB/s vs 网络接口 10-100Gbps
   - **更低延迟**：微秒级 vs 毫秒级
   - **自动检测**：NCCL会自动选择最佳通信路径
   - **避免冲突**：不需要手动配置网络接口

### 验证结果
通过以下命令验证NCCL初始化成功：
```bash
MASTER_ADDR=localhost MASTER_PORT=12345 NCCL_DEBUG=INFO /home/shouwei/.conda/envs/loongserve/bin/python -c "import torch; import torch.distributed as dist; dist.init_process_group('nccl', init_method='env://', rank=0, world_size=1); print('NCCL initialized successfully')"
```

### 经验总结
1. **系统环境差异**：不同集群的网络接口配置可能不同
2. **NVLink优先**：有NVLink时应该优先使用，而不是网络接口
3. **自动检测**：让NCCL自动选择最佳通信路径，避免手动配置错误
4. **调试方法**：使用 `NCCL_DEBUG=INFO` 查看NCCL的详细日志