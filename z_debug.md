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