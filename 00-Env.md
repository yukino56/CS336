# 环境配置

## Conda

创建环境

```bash
conda create -n myenv python=3.10
```

- `-n myenv` 环境名

激活 / 退出

```bash
conda activate myenv
conda deactivate
```

安装包

```bash
conda install pytorch torchvision torchaudio -c pytorch
```

查看环境与包

```bash
$ conda env list
# conda environments:
#
base                   /opt/anaconda3
d2l                    /opt/anaconda3/envs/d2l
d2l_env                /opt/anaconda3/envs/d2l_env
rnn_env                /opt/anaconda3/envs/rnn_env

$ conda list
```

删除环境

```bash
conda remove -n myenv --all
```

导出环境配置

```bash
conda env export > environment.yml
```

把当前激活的 python 解释器注册成 Jupyter kernel, 起名为 mien

```bash
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

## device

PyTorch 区分 device:

| device | existence  |
| ------ | ---------- |
| cpu    |            |
| cuda   | NVIDIA GPU |
| mps    | Apple GPU  |

