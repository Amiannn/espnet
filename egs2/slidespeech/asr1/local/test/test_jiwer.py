import jiwer

def read_file_(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in frs:
            data = fr.replace('\n', '').lower()
            uid, text = data.split(' ', 1)
            datas.append(text)
    return datas

def read_file(path):
    datas = []
    with open(path, 'r', encoding='utf-8') as frs:
        for fr in frs:
            data = fr.replace('\n', '').lower()
            text, uid = data.split('\t')
            datas.append(text)
    return datas

ref_path="/mnt/storage1/experiments/espnet/egs2/slidespeech/asr1/exp/asr_hybird_whisper_ctc/decode_asr_whisper_hybrid_asr_model_20epoch/test/score_wer/ref.trn"
hyp_path="/mnt/storage1/experiments/espnet/egs2/slidespeech/asr1/exp/asr_hybird_whisper_ctc/decode_asr_whisper_hybrid_asr_model_20epoch/test/score_wer/hyp.trn"
# hyp_path="/mnt/storage1/experiments/espnet/egs2/slidespeech/asr1/exp/asr_train_conformer_raw_en_bpe5000_sp_suffix/decode_asr_bs10_asr_model_valid.acc.ave_10best/test/text"

hyp_path = "./dump/raw/test/text"

refs = read_file(ref_path)
hyps = read_file_(hyp_path)

refs_filter, hyps_filter = [], []

for ref, hyp in zip(refs, hyps):
    if len(ref) == 0:
        continue
    refs_filter.append(ref)
    hyps_filter.append(hyp)

print(f'refs: {refs_filter[:3]}')
print(f'hyps: {hyps_filter[:3]}')

error = jiwer.wer(hyps_filter, refs_filter)

out = jiwer.process_words(
    refs_filter,
    hyps_filter,
)

print(jiwer.visualize_alignment(out))