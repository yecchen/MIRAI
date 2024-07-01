import os
import sys
import paramiko
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Execute commands on remote hosts via SSH.')
    parser.add_argument('--output_directory', default='../data/text_raw', help='Directory for output files.')
    parser.add_argument('--log_directory', default='../data/text_raw_log', help='Directory for log files.')
    parser.add_argument('--hosts', required=True, help='Comma-separated list of hostnames.')
    parser.add_argument('--username', required=True, help='SSH username.')
    parser.add_argument('--password', required=True, help='SSH password.')
    parser.add_argument('--project_dir', required=True, help='Project directory on the remote host.')
    parser.add_argument('--conda_path', required=True, help='Path to Conda installation on the remote host.')
    parser.add_argument('--env_name', required=True, help='Name of the Conda environment to activate.')
    parser.add_argument('--script_path', required=True, help='Path to the Python script to execute.')
    return parser.parse_args()

def main():
    args = parse_arguments()
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)
    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    cmd_template = (
        "nohup bash -c 'cd {project_dir} && "
        "export PATH=\"{conda_path}:$PATH\" && "
        "source {conda_path}/etc/profile.d/conda.sh && "
        "conda activate {env_name} && "
        "python {script_path} --total_processes {total_processes} --current_process {current_process} "
        "--output_db {output_dir}/output_db_{host_id}_{proc_id}.sqlite' "
        "&> {log_dir}/log_{host_id}_{proc_id}.txt &"
    )

    processes_per_host = 8
    hosts = args.hosts.split(',')
    for host_id, host in enumerate(hosts, start=1):
        try:
            ssh.connect(hostname=host, username=args.username, password=args.password)
            for proc_id in range(1, processes_per_host + 1):
                total_processes = len(hosts) * processes_per_host
                current_process = (host_id - 1) * processes_per_host + proc_id
                command = cmd_template.format(
                    project_dir=args.project_dir,
                    conda_path=args.conda_path,
                    env_name=args.env_name,
                    script_path=args.script_path,
                    total_processes=total_processes,
                    current_process=current_process,
                    host_id=host_id,
                    proc_id=proc_id,
                    output_dir=args.output_directory,
                    log_dir=args.log_directory
                )
                ssh.exec_command(command)
                print(f"Command executed on {host} for process {proc_id}: {command}")
        except Exception as e:
            print(f"Failed to connect or execute on {host}: {str(e)}")

    ssh.close()

if __name__ == '__main__':
    main()
    # python distribute_download_text.py --hosts "host1,host2" --username "your_username" --password "your_password" \
    # --project_dir "/remote/project/directory" --conda_path "/remote/conda/path" --env_name "remote_conda_environment" \
    # --script_path "/remote/script/path.py" --output_directory "/path/to/output" --log_directory "/path/to/logs"