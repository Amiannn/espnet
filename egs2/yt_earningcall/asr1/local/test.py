import os
import subprocess

script_path = "/share/nas165/amian/experiments/speech/tcpgen/espnet/egs2/institutional_investors/asr1/local/test.sh"
def get_fbank_pitch():
    command = f"bash {script_path}"
    output  = subprocess.run(
        command, 
        shell=True, 
        capture_output=True
    ).stdout.decode()

    if "Succeeded creating filterbank and pitch features for" in output:
        return 'success'
    return 'faild'

print(get_fbank_pitch())