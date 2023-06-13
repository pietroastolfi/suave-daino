import sys
import os
import subprocess
import argparse
from datetime import datetime
import inspect
import stat

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="suave")
parser.add_argument("--experiment_dir", type=str, default=None)
parser.add_argument("--base_experiment_dir", type=str, default="./experiments")
parser.add_argument("--data_dir", type=str, default="/data/datasets/imagenet")
parser.add_argument("--host", type=str, default=None)
parser.add_argument("--num_gpus", type=int, default=2)
parser.add_argument("--num_nodes", type=int, default=1)
parser.add_argument("--hours", type=int, default=80)
parser.add_argument("--mode", type=str, choices=["local", "slurm"], default="local")
parser.add_argument("--method", type=str, choices=["suave", "daino"], default="suave")
parser.add_argument("--script_file", type=str, default="train_10perc.sh")
parser.add_argument("--master_port", type=int, default=40001)

args = parser.parse_args()

# create experiment directory
root_dir = os.path.abspath(".")
if args.experiment_dir is None:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args.experiment_dir = f"{timestamp}-{args.name}"
full_experiment_dir = os.path.join(
    root_dir, args.base_experiment_dir, args.experiment_dir)
os.makedirs(full_experiment_dir, exist_ok=True)
print(f"Experiment directory: {full_experiment_dir}")
os.system(f"cp {args.script_file} {full_experiment_dir}/.")

# build training command
train_command = " ".join(r.strip() for r in open(args.script_file).readlines()).replace("\\", "")

if args.mode == "local":
    command = inspect.cleandoc(
        f"""
        torchrun \
        --nproc_per_node {args.num_gpus} \
        --nnodes 1 \
        --master_addr 127.0.0.1 \
        --master_port {args.master_port} \
        {args.method}/{train_command} \
        --data_path {args.data_dir} \
        --output_dir {full_experiment_dir}
        """
    )

# build slurm command
if args.mode == "slurm":
    assert args.host is not None

    command = inspect.cleandoc(
        f"""
        #!/bin/bash
        # SLURM SUBMIT SCRIPT
        #SBATCH -p {args.host}
        #SBATCH --job-name {args.name}
        #SBATCH --nodes {args.num_nodes}
        #SBATCH --ntasks-per-node {args.num_gpus}
        #SBATCH --gres=gpu:{args.num_gpus}
        #SBATCH -c 8
        #SBATCH -t {args.hours}
        #SBATCH --mem 80000
        #SBATCH -o {full_experiment_dir}/slurm-%j.out
        #SBATCH -e {full_experiment_dir}/slurm-%j.err

        master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
        export MASTER_ADDR=$master_addr
        export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
        
        # cancel job in case of error
        set -euo pipefail
        
        # Activate a conda environment:
        eval "$(conda shell.bash hook)"
        conda activate suavedaino-env

        cd {root_dir}

        # run training
        srun python {args.method}/{train_command} \
        --data_path {args.data_dir} \
        --output_dir {full_experiment_dir}
        """
    )

# write command
command_path = os.path.join(full_experiment_dir, "command.sh")
with open(command_path, "w") as f:
    f.write(command)

# add execution permission
st = os.stat(command_path)
os.chmod(command_path, st.st_mode | stat.S_IEXEC)

# run command
if args.mode in ["slurm"]:
    p = subprocess.Popen(f"sbatch {command_path}", shell=True, stdout=sys.stdout, stderr=sys.stdout)
    p.wait()
else:
    os.system(command_path)
