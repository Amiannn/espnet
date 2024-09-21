exp_folder=./exp/asr_whisper/run_medium_dotproduct_contextual_retriever_balanced_alpha0.8_suffix/decode_asr_whisper_ctc_greedy_c100_asr_model_valid.loss.ave_10best/test
# exp_folder=./exp/asr_whisper/run_medium_xdotproduct_contextual_retriever_balanced_alpha0.8_suffix/decode_asr_whisper_ctc_greedy_c100_asr_model_valid.loss.ave_10best/test
# exp_folder=./exp/asr_whisper/run_medium_lateinteraction_contextual_retriever_balanced_alpha0.8_suffix/decode_asr_whisper_ctc_greedy_c100_asr_model_valid.loss.ave_10best/test
# exp_folder=./exp/asr_whisper/run_medium_multilateinteraction_contextual_retriever_balanced_alpha0.8_suffix/decode_asr_whisper_ctc_greedy_c100_asr_model_valid.loss.ave_10best/test

python3 -m pyscripts.contextual.error_analysis.zh.caluate_context_retrieval_errors \
    --context_list_path "./local/contextual/rarewords/rareword_f10_test.txt" \
    --ref_context_path "./dump/raw/test/uttblist" \
    --hyp_context_path "${exp_folder}/context_text" \
    --hyp_context_prob_path "${exp_folder}/context_score" \
    --context_candidate_path "${exp_folder}/context_candidate" \
    --k 10
