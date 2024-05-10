import os

from tqdm  import tqdm
from pyscripts.utils.fileio import read_file
from pyscripts.utils.fileio import read_json
from pyscripts.utils.fileio import read_pickle
from pyscripts.utils.fileio import write_file
from pyscripts.utils.fileio import write_json
from pyscripts.utils.fileio import write_pickle

if __name__ == '__main__':
    ref_path = './dump/raw/test_clean/text'
    hyp_path = './exp/asr_whisper_medium_prompt_finetune/decode_asr_whisper_noctc_greedy_prompt_asr_model_valid.acc.ave_3best/test_clean/text'

    hyp_texts = [[d[0], " ".join(d[1:])] for d in read_file(hyp_path, sp=' ')]

    results   = []
    for uid, hyp_text in hyp_texts:
        _, hyp_text = hyp_text.split("OKAY THEN I'LL CONTINUE.")
        
        results.append([uid, hyp_text])
    
    output_path = './exp/asr_whisper_medium_prompt_finetune/decode_asr_whisper_noctc_greedy_prompt_asr_model_valid.acc.ave_3best/test_clean/'
    write_file(f'{output_path}/text_', results, sp=' ')
