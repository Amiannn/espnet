import os

def read_file(file_path, sp=' '):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().split(sp) for line in lines]

def write_file(datas, path, sp=' '):
    with open(path, 'w', encoding='utf-8') as fr:
        for data in datas:
            fr.write(sp.join([str(d) for d in data]) + "\n")

def process_datas(datas):
    new_datas = {}
    for data in datas:
        id = data[0]
        data = data[1:]
        ents = "".join(data).split(', ')
        new_datas[id] = ents
    return new_datas

def ent2idx(datas, ent_datas):
    new_datas = []
    new_probs = []
    for id in datas:
        ents = datas[id]
        ent_idxs  = []
        ent_probs = []
        for ent in ents:
            if ent == '':
                continue
            try:
                ent_idxs.append(str(ent_datas.index(ent)))
                ent_probs.append(str(0.99))
            except:
                print(f'out: {ent}')
        new_datas.append([id] + ent_idxs)
        new_probs.append([id] + ent_probs)
    return new_datas, new_probs

if __name__ == '__main__':
    file_path = "./exp/test/n_gram_matching/matched_entities_whisper_ctc_esun.txt"
    datas = read_file(file_path, '\t')
    datas = process_datas(datas)

    entity_path = "./local/contextual/rarewords/esun_earningcall.entity.txt"
    ent_datas = [d[0] for d in read_file(entity_path, '\t')]

    datas, probs = ent2idx(datas, ent_datas)
    
    output_path = "./exp/test/n_gram_matching/matched_entities_whisper_ctc_esun_idxs.txt"
    write_file(datas, output_path)

    output_path = "./exp/test/n_gram_matching/matched_entities_whisper_ctc_esun_probs.txt"
    write_file(probs, output_path)