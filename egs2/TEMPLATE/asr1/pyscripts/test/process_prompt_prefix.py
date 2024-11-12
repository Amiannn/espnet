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
    hyp_path = './hyp.txt'

    hyp_datas = read_file(hyp_path, sp=' ')
    hyp_datas = [[d[0], (" ".join(d[1:])).split('開始吧.')[-1]] for d in hyp_datas]
    print(hyp_datas[:10])

    output_path = hyp_path.replace('.txt', '_remove_prompt.txt')
    write_file(hyp_datas, output_path, sp=' ')