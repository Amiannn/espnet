import os

input_dir = './dump/raw'

def read_text(file_path):
    with open(file_path, 'r') as frs:
        datas = []
        for fr in frs:
            data = fr.replace("\n", "").split(' ')
            uid, text = data[0], " ".join(data[1:])
            if 'sp' in uid:
                continue
            datas.append([uid, text])
    return datas

def read_utt2spk(file_path):
    with open(file_path, 'r') as frs:
        datas = []
        for fr in frs:
            data = fr.replace("\n", "").split(' ')
            utt, spk = data[0], " ".join(data[1:])
            if 'sp' in utt:
                continue
            datas.append([utt, spk])
    return datas

def read_spk2utt(file_path):
    with open(file_path, 'r') as frs:
        datas = []
        for fr in frs:
            data = fr.replace("\n", "").split(' ')
            spk, utt = data[0], " ".join(data[1:])
            if 'sp' in utt:
                continue
            datas.append([spk, utt])
    return datas

def write_file(datas, output_path):
    with open(output_path, 'w') as fr:
        for data in datas:
            fr.write(" ".join(data) + '\n')

if __name__ == "__main__":
    folders = ['train_sp', 'dev', 'test']
    
    for folder in folders:
        data_folder = os.path.join(input_dir, folder)

        text_path    = os.path.join(data_folder, 'text')
        utt2spk_path = os.path.join(data_folder, 'utt2spk')
        spk2utt_path = os.path.join(data_folder, 'spk2utt')
        wavscp_path  = os.path.join(data_folder, 'wav.scp')

        text_data    = read_text(text_path)
        utt2spk_data = read_utt2spk(utt2spk_path)
        spk2utt_data = read_spk2utt(spk2utt_path)

        wav_path_root = './data/wav/'

        folder_name = folder.replace('_sp', '')
        wavscp_data  = [[
            utt, 
            os.path.join(wav_path_root, folder_name, f'{utt}.wav')
            ] for utt, _ in text_data
        ]

        output_folder = os.path.join('./data', folder_name)
        os.makedirs(output_folder, exist_ok=True)

        text_output_path = os.path.join(output_folder, 'text')
        write_file(text_data, text_output_path)

        utt2spk_output_path = os.path.join(output_folder, 'utt2spk')
        write_file(utt2spk_data, utt2spk_output_path)

        spk2utt_output_path = os.path.join(output_folder, 'spk2utt')
        write_file(spk2utt_data, spk2utt_output_path)

        wavscp_output_path = os.path.join(output_folder, 'wav.scp')
        write_file(wavscp_data, wavscp_output_path)