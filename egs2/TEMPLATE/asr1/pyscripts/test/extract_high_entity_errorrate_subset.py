import os

def read_file(file_path, sp=' '):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().split(sp) for line in lines]

def write_file(datas, path, sp=' '):
    with open(path, 'w', encoding='utf-8') as fr:
        for data in datas:
            fr.write(sp.join([str(d) for d in data]) + "\n")

if __name__ == '__main__':
    error_pattern_path = "../asr1/exp/asr_whisper_medium_lora_decoder/decode_asr_whisper_noctc_greedy_asr_model_3epoch/test/analysis/error_patterns.tsv"
    error_pattern_datas = read_file(error_pattern_path, sp='\t')[1:]

    error_pattern_datas = [d[0] for d in error_pattern_datas if float(d[3]) >= 50]


    entity_path = "./local/contextual/rarewords/esun_earningcall.entity.txt"
    entity_datas = [d[0] for d in read_file(entity_path, sp='\t')]

    error_pattern_idxs = [entity_datas.index(d) for d in error_pattern_datas if d in entity_datas]

    uttblist_idx_path = "./dump/raw/test/uttblist_idx_entity_earningcall"
    uttblist_datas = {d[0]: [int(idx) for idx in d[1:]] for d in read_file(uttblist_idx_path, sp=' ')}

    sub_testset = []
    for uttb, idxs in uttblist_datas.items():
        for idx in idxs:
            if idx in error_pattern_idxs:
                sub_testset.append(uttb)
    
    sub_testset = sorted(list(set(sub_testset)))

    wav_scp_path = "./dump/raw/test/wav.scp"
    wav_scp_datas = {d[0]: d[1] for d in read_file(wav_scp_path, sp=' ')}

    sub_wav_scp_datas = [[uttb, wav_scp_datas[uttb]] for uttb in sub_testset]

    output_dir = "./dump/raw/test_small"
    os.makedirs(output_dir, exist_ok=True)
    write_file(sub_wav_scp_datas, os.path.join(output_dir, 'wav.scp'), sp=' ')

    text_datas = {d[0]: " ".join(d[1:]) for d in read_file("./dump/raw/test/text", sp=' ')}
    sub_text_datas = [[uttb, text_datas[uttb]] for uttb in sub_testset]
    write_file(sub_text_datas, os.path.join(output_dir, 'text'), sp=' ')

    sub_uttblist_datas = {uttb: uttblist_datas[uttb] for uttb in sub_testset}
    sub_uttblist_datas = [[uttb] + idxs for uttb, idxs in sub_uttblist_datas.items()]
    write_file(sub_uttblist_datas, os.path.join(output_dir, 'uttblist_idx_entity_earningcall'), sp=' ')