import os

from pyscripts.contextual.error_analysis.zh.caluate_context_retrieval_errors import main as error_analysis_func

def read_file(file_path, sp=' '):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return [line.strip().split(sp) for line in lines]

def write_file(datas, path, sp=' '):
    with open(path, 'w', encoding='utf-8') as fr:
        for data in datas:
            fr.write(sp.join([str(d) for d in data]) + "\n")

def select_max(idxs, probs):
    new_idxs, new_probs = [], []
    for idx, prob in zip(idxs, probs):
        result = {}
        new_idx, new_prob = [], []
        for i in range(len(idx)):
            index = idx[i]
            result[index] = result[index] + [prob[i]] if index in result else [prob[i]]
        for idx in result:
            prob = max(result[idx])
            new_idx.append(idx)
            new_prob.append(prob)
        new_idxs.append(new_idx)
        new_probs.append(new_prob)
    return new_idxs, new_probs

def merge(idxs_a, probs_a, idxs_b, probs_b):
    idxs_m, probs_m = [], []
    for idx_a, prob_a, idx_b, prob_b in zip(idxs_a, probs_a, idxs_b, probs_b):
        idxs_m.append(idx_a + idx_b)
        probs_m.append(prob_a + prob_b)
    return idxs_m, probs_m

if __name__ == '__main__':
    context_list_path      = "./local/contextual/rarewords/esun_earningcall.entity.txt"
    
    enc_context_idx_path   = "./exp/asr_whisper/run_medium_xdotproduct_contextual_retriever_suffix/decode_asr_whisper_ctc_greedy_c100_entity_earningcall_asr_model_valid.loss.ave_10best/test/context_idx"
    enc_context_score_path = "./exp/asr_whisper/run_medium_xdotproduct_contextual_retriever_suffix/decode_asr_whisper_ctc_greedy_c100_entity_earningcall_asr_model_valid.loss.ave_10best/test/context_score"
    dec_context_idx_path   = "./exp/asr_whisper/run_medium_contextual_adapter_decoder/decode_asr_whisper_contextual_adapter_decoder_c100_entity_earningcall_asr_model_valid.loss.ave_10best_fixed/test/context_idx"
    dec_context_score_path = "./exp/asr_whisper/run_medium_contextual_adapter_decoder/decode_asr_whisper_contextual_adapter_decoder_c100_entity_earningcall_asr_model_valid.loss.ave_10best_fixed/test/context_score"
    
    context_list_datas     = [d[0] for d in read_file(context_list_path, sp=' ')]
    
    uid_datas              = [[d[0]] for d in read_file(enc_context_idx_path, sp=' ')]
    enc_context_idx        = [list(map(int, d[1:])) for d in read_file(enc_context_idx_path, sp=' ')]
    enc_context_prob_datas = [list(map(float, d[1:])) for d in read_file(enc_context_score_path, sp=' ')]

    dec_context_idx        = [list(map(int, d[1:])) for d in read_file(dec_context_idx_path, sp=' ')]
    dec_context_prob_datas = [list(map(float, d[1:])) for d in read_file(dec_context_score_path, sp=' ')]
    
    dec_context_idx, dec_context_prob_datas = select_max(dec_context_idx, dec_context_prob_datas)

    merge_context_idx, merge_contetx_prob_datas = merge(
        enc_context_idx,
        enc_context_prob_datas,
        dec_context_idx,
        dec_context_prob_datas,
    )

    merge_context_idx, merge_contetx_prob_datas = select_max(merge_context_idx, merge_contetx_prob_datas)
    merge_context_idx, merge_contetx_prob_datas = merge(
        uid_datas,
        uid_datas,
        dec_context_idx,
        dec_context_prob_datas,
    )

    output_root = "./exp/test"
    output_idx_path = os.path.join(output_root, 'context_idx')
    write_file(merge_context_idx, output_idx_path)

    output_prob_path = os.path.join(output_root, 'context_score')
    write_file(merge_contetx_prob_datas, output_prob_path)

    error_analysis_func(
        context_list_path="./local/contextual/rarewords/esun_earningcall.entity.txt",
        ref_context_path="./dump/raw/test/uttblist_idx_entity_earningcall",
        hyp_context_path=enc_context_idx_path,
        hyp_context_prob_path=enc_context_score_path,
        # hyp_context_path=dec_context_idx_path,
        # hyp_context_prob_path=dec_context_score_path,
        # hyp_context_path=output_idx_path,
        # hyp_context_prob_path=output_prob_path,
        context_candidate_path="./exp/asr_whisper/run_medium_contextual_adapter_decoder/decode_asr_whisper_contextual_adapter_decoder_c100_entity_earningcall_asr_model_valid.loss.ave_10best_fixed/test/context_candidate",
        k=10,
        thres=0.5,
    )