import os

def read_file(file_path, sp=" "):
    with open(file_path, "r") as f:
        lines = f.readlines()
        lines = [line.strip().split(sp) for line in lines]
    return lines

def write_file(file_path, lines, sp=" "):
    with open(file_path, "w") as f:
        for line in lines:
            f.write(sp.join(line) + "\n")

ref_path = "./dump/raw/dev/text"
ocr_path = "./local/contextual/metadata/related_files/dev/keywords"

if __name__ == '__main__':
    uttids = [d[0] for d in read_file(ref_path)]
    
    datas  = [[uid, uid.split('_')[0]] for uid in uttids]
    write_file("./dump/raw/dev/wav2session_domains", datas)

    # ocr_datas = [[d[0].replace('-', '_'), "$".join(d[1:])] for d in read_file(ocr_path)]
    # write_file("dump/keywords", ocr_datas)
