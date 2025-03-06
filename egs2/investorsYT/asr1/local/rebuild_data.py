import os
import kaldiio
import soundfile as sf
from concurrent.futures import ProcessPoolExecutor

def read_file(file_path):
    with open(file_path, 'r') as f:
        return [line.strip().split() for line in f]

def process_utterance(data, output_folder):
    utt_id, wav_path = data
    if 'sp' in utt_id:
        return
    sample_rate, waveform = kaldiio.load_mat(wav_path)
    output_path = os.path.join(output_folder, f"{utt_id}.wav")
    sf.write(output_path, waveform, sample_rate)
    print(f"{output_path} saved")

def process_folder(folder):
    wav_scp_path = os.path.join('dump', 'raw', folder, 'wav.scp')
    datas = read_file(wav_scp_path)
    
    # Create the output folder once per folder
    output_folder = os.path.join('data', 'wav', folder.replace('_sp', ''))
    os.makedirs(output_folder, exist_ok=True)
    
    # Optionally process a subset, here using the first 10 entries
    for data in datas:
        process_utterance(data, output_folder)

if __name__ == "__main__":
    folders = ['train_sp', 'dev', 'test']
    # Use a process pool to process each folder concurrently.
    with ProcessPoolExecutor() as executor:
        executor.map(process_folder, folders)
