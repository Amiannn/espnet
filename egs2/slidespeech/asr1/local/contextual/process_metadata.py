import os

def read_file(file_path, sp=' '):
    with open(file_path, 'r') as f:
        return [line.strip().split(sp) for line in f]

def write_file(file_path, data, sp=' '):
    with open(file_path, 'w') as f:
        for d in data:
            f.write(sp.join(d) + '\n')

if __name__ == '__main__':
    metadata_S95_path  = './local/contextual/metadata/related_files/S95/keywords'
    metadata_dev_path  = './local/contextual/metadata/related_files/dev/keywords'
    metadata_test_path = './local/contextual/metadata/related_files/test/keywords'

    output_dir = './local/contextual/contexts'

    metadata_train = read_file(metadata_S95_path)
    metadata_dev   = read_file(metadata_dev_path)
    metadata_test  = read_file(metadata_test_path)
    
    metadata = metadata_train + metadata_dev

    keywords = []
    for m in metadata:
        if len(m) < 2:
            continue
        keywords += m[1].split('$')

    keywords = [[k] for k in sorted(list(set(keywords)), key=lambda x: len(x), reverse=True)]
    output_path = os.path.join(output_dir, 'context_keywords_train.txt')
    write_file(output_path, keywords)

    keywords = []
    for m in metadata_test:
        if len(m) < 2:
            continue
        keywords += m[1].split('$')

    keywords = [[k] for k in sorted(list(set(keywords)), key=lambda x: len(x), reverse=True)]
    output_path = os.path.join(output_dir, 'context_keywords_test.txt')
    write_file(output_path, keywords)