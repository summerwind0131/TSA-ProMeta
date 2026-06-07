# Cursor Remote SSH 集群运行指南

这份说明假设代码在集群路径：

```text
/mnt/hpc/home/lihan/fengyuan/task_prometa_code/proMeta
```

老师提供的数据在：

```text
/mnt/hpc/home/lihan/fengyuan/ProMeta/data
```

## 1. 在 Cursor 打开远程项目

1. Cursor 里连接 `tmu-hpc`。
2. 打开远程目录：

```text
/mnt/hpc/home/lihan/fengyuan/task_prometa_code/proMeta
```

3. 在 Cursor 文件树里进入：

```text
official_ProMeta/ProMeta
```

你应该能看到：

```text
slurm_smoke.slurm
slurm_benchmark_array.slurm
run_benchmark.sh
summarize_benchmark.py
```

## 2. 先检查数据

在 Cursor 的远程终端运行：

```bash
ls -lh /mnt/hpc/home/lihan/fengyuan/ProMeta/data
ls -lh /mnt/hpc/home/lihan/fengyuan/ProMeta/data/out
```

`data/out` 里需要有 6 个 pkl：

```text
term2pre_cases_train.pkl
term2pre_controls_train.pkl
term2pre_cases_valid.pkl
term2pre_controls_valid.pkl
term2pre_cases_test.pkl
term2pre_controls_test.pkl
```

## 3. 准备 conda 环境

```bash
source /mnt/hpc/home/lihan/miniconda3/etc/profile.d/conda.sh
conda activate ProMeta
```

如果环境不存在，先在：

```bash
cd /mnt/hpc/home/lihan/fengyuan/task_prometa_code/proMeta/official_ProMeta
conda env create -f environment.yml
```

然后检查：

```bash
python -c "import torch, torchmetrics, pandas, numpy, sklearn, matplotlib; print(torch.cuda.is_available())"
```

## 4. 先提交 smoke test

不要直接在登录节点运行 `bash run_benchmark.sh`。

在 Cursor 远程终端运行：

```bash
cd /mnt/hpc/home/lihan/fengyuan/task_prometa_code/proMeta/official_ProMeta/ProMeta
sbatch --test-only slurm_smoke.slurm
sbatch slurm_smoke.slurm
```

查看队列：

```bash
squeue -u lihan
```

查看日志，把 `JOBID` 换成实际编号：

```bash
tail -f /mnt/hpc/home/lihan/fengyuan/task_prometa_output/logs/prometa_smoke_JOBID.out
```

smoke test 成功后应有：

```bash
ls /mnt/hpc/home/lihan/fengyuan/task_prometa_output/checkpoints/support_4
ls /mnt/hpc/home/lihan/fengyuan/task_prometa_output/benchmark_results/support_4
ls /mnt/hpc/home/lihan/fengyuan/task_prometa_output/benchmark_summary
```

## 5. 正式提交完整实验

确认 smoke test 成功后：

```bash
cd /mnt/hpc/home/lihan/fengyuan/task_prometa_code/proMeta/official_ProMeta/ProMeta
sbatch --test-only slurm_benchmark_array.slurm
sbatch slurm_benchmark_array.slurm
```

默认会跑：

```text
shot: 4, 8, 16, 32
seed: 42, 43, 44, 45, 46
```

`#SBATCH --array=0-19%4` 表示总共 20 个任务，最多同时跑 4 个。  
如果想更保守，把 `slurm_benchmark_array.slurm` 里的 `%4` 改成 `%2`。

## 6. 汇总结果

等全部任务结束后：

```bash
cd /mnt/hpc/home/lihan/fengyuan/task_prometa_code/proMeta/official_ProMeta/ProMeta
source /mnt/hpc/home/lihan/miniconda3/etc/profile.d/conda.sh
conda activate ProMeta
python summarize_benchmark.py --output_dir /mnt/hpc/home/lihan/fengyuan/task_prometa_output
```

结果在：

```text
/mnt/hpc/home/lihan/fengyuan/task_prometa_output/benchmark_summary
```

重点看：

```text
summary_metrics.csv
paired_task_delta.csv
statistical_tests.csv
shot_curve_auroc.png
shot_curve_auprc.png
```
