import os

from tqdm import tqdm
from pyscripts.contextual.utils.dataio import read_file
from pyscripts.contextual.utils.dataio import read_json
from pyscripts.contextual.utils.dataio import write_file

TEST_BLIST_PATH = "./local/contextual/rarewords/all_rare_words.txt"
HYP_PROMPT_PATH = "./exp/asr_whisper/tune_medium__enc_conv2_xphone__ga_warmup__mediumbatch/debug/7epoch/predicts_200.json"
BLIST_IDX_PATH  = "./dump/raw/test_clean/uttblist_idx"

PROMPT_TEMPLATE         = '''The topic of today's speech is {}. Okay then I'll continue.'''
PROMPT_NON_ENT_TEMPLATE = '''Okay then I'll continue.'''

if __name__ == '__main__':
    datas_path = './dump/raw'
    path       = os.path.join(datas_path, 'test_clean')
    blist = [b[0] for b in read_file(TEST_BLIST_PATH, sp=' ')]

    print(f'processing {path}...')
    blist_datas = [[b[0], b[1:]] for b in read_file(BLIST_IDX_PATH, sp=' ')]
    hyp_prompts = read_json(HYP_PROMPT_PATH)

    rareword_datas = []
    for uid, blist_idx in tqdm(blist_datas):
        idxs     = [int(idx) for idx in blist_idx if idx != ""]
        hyp_idxs = [idxs[i - 1] for i in hyp_prompts[uid]['pred'] if (i - 1) < len(idxs)]
        
        uttblist = [blist[idx] for idx in hyp_idxs]
        print(f'{uid}, {uttblist}')

        if len(uttblist) > 0:
            prompt = PROMPT_TEMPLATE.format(", ".join(uttblist))
        else:
            prompt = PROMPT_NON_ENT_TEMPLATE
        rareword_datas.append([uid, prompt.upper()])

    output_path_uttblist = os.path.join(path, 'prompt_')
    write_file(output_path_uttblist, rareword_datas)