import argparse
import subprocess
import sys

def slurm_submit(script):
    """
    Submit the SLURM script using sbatch and return the job ID.
    """
    try:
        # Use a list for the command and pass the script via stdin
        output = subprocess.check_output(["sbatch"], input=script, universal_newlines=True)
        job_id = output.strip().split()[-1]
        return job_id
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {e.output}", file=sys.stderr)
        sys.exit(1)

def submit(
        conda_env_name, 
        script_name,
        num_gpus, 
        partition, 
        job_name, 
        mem, 
        cpus, 
        time, 
        note, 
        dataset,
        training, 
        paths, 
        load_jobid, 
        resume_jobid,
        gpu_type, 
        override
        ):
    """
    Construct and submit the SLURM script with the specified parameters.
    """

    """Submit job to cluster."""
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}    
#SBATCH --qos={partition}
#SBATCH --account=portia
#SBATCH --time={time}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus*num_gpus}
#SBATCH --gpus={num_gpus}
#SBATCH --mem={mem}G
#SBATCH --verbose  
#SBATCH --open-mode=append
#SBATCH -o ./OutFiles/slurm-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=eabe@uw.edu
module load conda
set -x
nvidia-smi
conda activate {conda_env_name}
echo $SLURMD_NODENAME
python -u ./scripts/{script_name}.py paths={paths} note={note} training={training} dataset={dataset} load_jobid={load_jobid} run_id={resume_jobid if resume_jobid else '$SLURM_JOB_ID'} {override}
            """
    print(f"Submitting job")
    print(script)
    job_id = slurm_submit(script)
    print(job_id)

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Submit a SLURM job with specified GPU type.')
    parser.add_argument('--conda_env_name', type=str, default='sphinx',
                        help='Name of the conda environment (default: sphinx)')
    parser.add_argument('--script_name', type=str, default='train_basic_imitation',
                        help='Name of the script to run (default: train_basic_imitation)')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='Number of GPUs to request (default: 1)')
    parser.add_argument('--gpu_type', type=str, default='all',
                        help='Type of GPU to request (default: all)')
    parser.add_argument('--job_name', type=str, default='sphinx',
                        help='Name of the SLURM job (default: sphinx)')
    parser.add_argument('--mem', type=int, default=200,
                        help='Memory in GB (default: 200)')
    parser.add_argument('--cpus', type=int, default=8,
                        help='Number of CPU cores (default: 8)')
    parser.add_argument('--time', type=str, default='1-00:00:00',
                        help='Time limit for the job day-hr-min-sec (default: 1-00:00:00)')
    parser.add_argument('--partition', type=str, default='normal',
                        help='Partition to run job (default: normal)')
    parser.add_argument('--note', type=str, default='hyak_ckpt',
                        help='Note for job (default: hyak_ckpt)')
    parser.add_argument('--dataset', type=str, default='imitation_walk_anipose_data_v1',
                        help='Name of dataset yaml  (default: imitation_walk_anipose_data_v1)')
    parser.add_argument('--training', type=str, default='ppo_basic_imitation_low_kl',
                        help='Name of training yaml  (default: ppo_basic_imitation_low_kl)')
    parser.add_argument('--paths', type=str, default='tillicum',
                        help='Name of paths yaml  (default: tillicum)')
    parser.add_argument('--load_jobid', type=str, default='',
                        help='JobID to load policy/rollout from (default: '')')
    parser.add_argument('--resume_jobid', type=str, default='',
                        help='JobID of a previous run to resume checkpoints from (calibrate_vnc_offline, train_closed_loop_cmaes) (default: '')')
    parser.add_argument('--override', type=str, default='',
                        help='Override parameters for the job (default: '')')

    args = parser.parse_args()

    submit(
        conda_env_name=args.conda_env_name,
        script_name=args.script_name,
        num_gpus=args.num_gpus,
        job_name=args.job_name,
        mem=args.mem,
        cpus=args.cpus,
        time=args.time,
        partition=args.partition,
        note=args.note,
        dataset=args.dataset,
        training=args.training,
        paths=args.paths,
        load_jobid=args.load_jobid,
        resume_jobid=args.resume_jobid,
        gpu_type=args.gpu_type,
        override=args.override,
    )

if __name__ == "__main__":
    main()
    
##### Saving commands #####
''' 
#### cancel all jobs: 
# squeue -u $USER -h | awk '{print $1}' | xargs scancel

#SBATCH --exclude=g[003,010,024,023]


#### wandb regex: ^(?!.*table)(?!.*std).*$|^reward*&

python ./scripts/tillicum_run.py --partition=normal --num_gpus=1 --script_name=train_basic_imitation

'''