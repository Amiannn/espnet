import os
import subprocess

from local.utils import read_file

dataset_root_path = "./dump/raw"
# dataset_root_path = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/esun/asr1/dump/raw"
# dataset_root_path = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/aishell/asr1/dump/raw"
dataset_folders   = ["train_sp", "dev", "test"]

def get_duration(path):
    command = f"sox {wav_path} -n stat 2>&1 | sed -n 's#^Length (seconds):[^0-9]*\([0-9.]*\)$#\\1#p'"
    output  = subprocess.run(
        command, 
        shell=True, 
        capture_output=True
    ).stdout.decode().replace('\n', '')
    output = float(output)
    return output

if __name__ == '__main__':
    sample_rate = 16000
    
    for folder in dataset_folders:
        utt2num_samples_path = os.path.join(dataset_root_path, folder, 'utt2num_samples')
        seconds = [int(d[1]) / sample_rate for d in read_file(utt2num_samples_path, sp=' ')]
        total_seconds = sum(seconds)
        print(f'{folder}: {int(total_seconds / 3600)}hr')