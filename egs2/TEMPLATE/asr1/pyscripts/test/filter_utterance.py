#!/usr/bin/env python3

import sys

def main():
    # Input file containing the utterance IDs (one per line, in the order you want).
    UTTERANCE_IDS = "/mnt/storage1/experiments/espnet/egs2/esun/asr1_contextual/exp/asr_whisper/run_medium_contextual_adapter_decoder_alpha0.9/decode_asr_whisper_contextual_adapter_decoder_c300_entity_earningcall_test_asr_model_valid.loss.ave_10best_fixed/test_small/text"
    # The file you want to filter.
    SOURCE_FILE = "/mnt/storage1/experiments/espnet/egs2/esun/asr1_contextual/exp/asr_whisper/run_medium_contextual_adapter_decoder_alpha0.9/decode_asr_whisper_contextual_adapter_decoder_c300_entity_earningcall_asr_model_valid.loss.ave_10best_fixed/test/text"
    # The file where we'll write the results.
    OUTPUT_FILE = "/mnt/storage1/experiments/espnet/egs2/esun/asr1_contextual/exp/asr_whisper/run_medium_contextual_adapter_decoder_alpha0.9/decode_asr_whisper_contextual_adapter_decoder_c300_entity_earningcall_asr_model_valid.loss.ave_10best_fixed/test_small/text"
    
    # Read all utterance IDs in order.
    with open(UTTERANCE_IDS, 'r', encoding='utf-8') as f:
        ids = [(line.strip()).split(' ')[0] for line in f if line.strip()]
    
    # Read big_file.txt into memory (list of lines).
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        big_file_lines = f.readlines()
    
    # Open the output file for writing.
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out:
        # For each ID (in the order they appear in utterance_ids.txt)...
        for u_id in ids:
            print(f'uid: {u_id}')
            # Scan through big_file.txt lines
            for line in big_file_lines:
                # Simple substring check: if 'u_id' is anywhere in 'line'
                # If you need an exact match or "whole word" match, see below.
                if u_id == line.split(' ')[0]:
                    out.write(line)
    
    print(f"Filtered lines saved to '{OUTPUT_FILE}' in the order of IDs from '{UTTERANCE_IDS}'.")

if __name__ == "__main__":
    main()
