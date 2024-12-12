import os
import jieba
import argparse
from collections import defaultdict

from tqdm import tqdm
from jiwer import cer, wer, mer

from pyscripts.utils.fileio import read_file, write_file
from pyscripts.utils.text_aligner import align_to_index


def is_ascii(string):
    """Check if a string contains only ASCII characters (English letters)."""
    try:
        string.encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def split_non_english_words(words):
    """
    Split non-English words into individual characters.
    Keep English words as they are.
    """
    processed = []
    for word in words:
        if is_ascii(word):
            processed.append(word)
        else:
            processed.extend(list(word))
    return processed


def concatenate_non_english_chars(words):
    """
    Concatenate consecutive non-English characters.
    Keep English words separated by spaces.
    """
    result = []
    temp_word = ""
    for word in words:
        if is_ascii(word):
            if temp_word:
                result.append(temp_word)
                temp_word = ""
            result.append(word)
        else:
            temp_word += word
    if temp_word:
        result.append(temp_word)
    return " ".join(result)


def is_phrase_in_sentence(segmented_phrase, segmented_sentence):
    phrase_len = len(segmented_phrase)
    for i in range(len(segmented_sentence) - phrase_len + 1):
        if segmented_sentence[i:i + phrase_len] == segmented_phrase:
            return True
    return False


def resegment_sentence(sentence):
    sentence_no_space  = sentence.replace(" ", "")
    segmented_sentence = list(jieba.cut(sentence_no_space))
    return segmented_sentence


def find_rare_words(sentence, entity_phrases):
    # Segment the sentence using jieba
    sentence_no_space  = sentence.replace(" ", "")
    segmented_sentence = list(jieba.cut(sentence_no_space))

    # Set to keep track of detected phrases
    detected_phrases = []

    # Check if each phrase is present in the segmented sentence
    for phrase in entity_phrases:
        segmented_phrase = list(jieba.cut("".join(phrase.split(' '))))
        segmented_phrase_str = " ".join(segmented_phrase)
        if is_phrase_in_sentence(segmented_phrase, segmented_sentence):
            if not any(segmented_phrase_str in detected for _, detected in detected_phrases):
                detected_phrases.append([None, phrase])
    detected_phrases = [d[-1] for d in detected_phrases]
    return detected_phrases


class ASREvaluator:
    def __init__(self, rare_words_list):
        self.rare_words = rare_words_list
        self.initialize_data()

    def initialize_data(self):
        """Initialize data structures for storing sentences and error patterns."""
        self.reference_sentences = []
        self.hypothesis_sentences = []
        self.ref_rareword_sentences = []
        self.hyp_rareword_sentences = []
        self.ref_common_sentences = []
        self.hyp_common_sentences = []
        self.ref_rare_english = []
        self.hyp_rare_english = []
        self.ref_rare_non_english = []
        self.hyp_rare_non_english = []
        self.error_patterns = defaultdict(lambda: defaultdict(int))
        self.rareword_counts = defaultdict(int)

    def process_utterance(self, reference, hypothesis):
        """
        Process a single reference and hypothesis pair.
        Updates internal data structures with alignment and error patterns.
        """
        ref_id, ref_words = reference
        hyp_id, hyp_words = hypothesis

        if not hyp_words:
            print(f"Error: Hypothesis for {ref_id} is empty!")
            return

        # Find rare words in the reference
        ref_sentence = " ".join(ref_words)
        rare_words_in_ref = find_rare_words(ref_sentence, self.rare_words)
        
        hyp_sentence = " ".join(hyp_words)
        
        ref_words = resegment_sentence(ref_sentence)
        hyp_words = resegment_sentence(hyp_sentence)

        # Align reference and hypothesis words
        alignment_chunks = align_to_index(ref_words, hyp_words)

        # Preprocess sentences
        ref_processed = split_non_english_words(ref_words)
        hyp_processed = split_non_english_words(hyp_words)
        self.reference_sentences.append(" ".join(ref_processed))
        self.hypothesis_sentences.append(" ".join(hyp_processed))

        # Initialize temporary lists for this utterance
        ref_rare_words = []
        hyp_rare_words = []
        ref_common_words = []
        hyp_common_words = []
        ref_rare_eng_words = []
        hyp_rare_eng_words = []
        ref_rare_non_eng_words = []
        hyp_rare_non_eng_words = []
        processed_hyp_indices = set()

        for ref_word, hyp_chunk, _, hyp_indices in alignment_chunks:
            ref_word_clean = ref_word.replace("-", "")
            hyp_word_combined = concatenate_non_english_chars(
                [w.replace("-", "") for w in hyp_chunk]
            )

            if ref_word_clean in rare_words_in_ref:
                # Update rare word counts and error patterns
                self.rareword_counts[ref_word_clean] += 1
                if ref_word_clean != hyp_word_combined:
                    self.error_patterns[ref_word_clean][hyp_word_combined] += 1

                ref_rare_words.append(ref_word_clean)
                hyp_rare_words.append(hyp_word_combined)

                if is_ascii(ref_word_clean):
                    ref_rare_eng_words.append(ref_word_clean)
                    hyp_rare_eng_words.append(hyp_word_combined)
                else:
                    ref_rare_non_eng_words.append(ref_word_clean)
                    hyp_rare_non_eng_words.append(hyp_word_combined)
            elif not processed_hyp_indices.intersection(hyp_indices):
                ref_common_words.append(ref_word_clean)
                hyp_common_words.append(hyp_word_combined)
                processed_hyp_indices.update(hyp_indices)

        # Append processed data for this utterance
        self.ref_rareword_sentences.append(
            ",".join(ref_rare_words) if ref_rare_words else "correct"
        )
        self.hyp_rareword_sentences.append(
            ",".join(hyp_rare_words) if hyp_rare_words else "correct"
        )
        self.ref_common_sentences.append(
            ",".join(ref_common_words) if ref_common_words else "correct"
        )
        self.hyp_common_sentences.append(
            ",".join(hyp_common_words) if hyp_common_words else "correct"
        )

        if ref_rare_eng_words:
            self.ref_rare_english.append(" ".join(ref_rare_eng_words))
            self.hyp_rare_english.append(" ".join(hyp_rare_eng_words))
        if ref_rare_non_eng_words:
            self.ref_rare_non_english.append(" ".join(ref_rare_non_eng_words))
            self.hyp_rare_non_english.append(" ".join(hyp_rare_non_eng_words))

    def finalize_sentences(self):
        """Clean up sentences by splitting non-English words into characters."""
        def clean(sentences):
            cleaned = []
            for sentence in sentences:
                tokens = split_non_english_words(sentence.split())
                cleaned.append(" ".join(tokens).strip())
            return cleaned

        self.ref_rareword_sentences = clean(self.ref_rareword_sentences)
        self.hyp_rareword_sentences = clean(self.hyp_rareword_sentences)
        self.ref_common_sentences = clean(self.ref_common_sentences)
        self.hyp_common_sentences = clean(self.hyp_common_sentences)

    def compute_metrics(self):
        """Compute MER, WER, and CER for the collected sentences."""
        self.finalize_sentences()
        self.overall_mer = mer(self.reference_sentences, self.hypothesis_sentences)
        self.rareword_mer = mer(
            self.ref_rareword_sentences, self.hyp_rareword_sentences
        )
        self.common_mer = mer(self.ref_common_sentences, self.hyp_common_sentences)
        try:
            self.rare_eng_wer = wer(self.ref_rare_english, self.hyp_rare_english)
        except:
            self.rare_eng_wer = 0
        self.rare_non_eng_cer = cer(
            self.ref_rare_non_english, self.hyp_rare_non_english
        )

        self.ref_rareword_sentences = [s.replace(' ', '') if s != "correct" else "" for s in self.ref_rareword_sentences]
        self.hyp_rareword_sentences = [s.replace(' ', '') if s != "correct" else "" for s in self.hyp_rareword_sentences]

        # Display metrics
        print(f"Overall MER: {self.overall_mer * 100:.2f}%")
        print(f"Rare Words MER: {self.rareword_mer * 100:.2f}%")
        print(f"Common Words MER: {self.common_mer * 100:.2f}%")
        print(f"Rare English Words WER: {self.rare_eng_wer * 100:.2f}%")
        print(f"Rare Non-English Words CER: {self.rare_non_eng_cer * 100:.2f}%")

    def save_results(self, uttids, word2idx, output_dir):
        """Write processed sentences and error patterns to files."""
        os.makedirs(output_dir, exist_ok=True)

        hyp_rare_idx   = []
        hyp_rare_score = []
        for hyp_sent, uid in zip(self.ref_rareword_sentences, uttids):
            if hyp_sent != "":
                hyps = list(set(hyp_sent.split(',')))
                idxs = " ".join([str(word2idx[hyp]) for hyp in hyps if hyp in word2idx])
                scores = " ".join([str(0.99) for hyp in hyps if hyp in word2idx])
            else:
                idxs = ""
                scores = ""
            hyp_rare_idx.append(f'{uid} {idxs}')            
            hyp_rare_score.append(f'{uid} {scores}')            
            
        # Define file mappings
        file_data = {
            "reference_sentences": self.reference_sentences,
            "hypothesis_sentences": self.hypothesis_sentences,
            "ref_common_sentences": self.ref_common_sentences,
            "hyp_common_sentences": self.hyp_common_sentences,
            "ref_rareword_sentences": self.ref_rareword_sentences,
            "hyp_rareword_sentences": self.hyp_rareword_sentences,
            "ref_rare_english": self.ref_rare_english,
            "hyp_rare_english": self.hyp_rare_english,
            "ref_rare_non_english": self.ref_rare_non_english,
            "hyp_rare_non_english": self.hyp_rare_non_english,
            # "hyp_context_idx": hyp_rare_idx,
            # "hyp_context_score": hyp_rare_score,
        }

        # Write sentences to files
        for filename, data in file_data.items():
            output_path = os.path.join(output_dir, filename)
            write_file(output_path, [[line] for line in data], sp="")

        # Write error patterns to a TSV file
        error_pattern_list = []
        for word, errors in self.error_patterns.items():
            total_errors = sum(errors.values())
            frequency = self.rareword_counts[word]
            error_rate = total_errors / frequency if frequency > 0 else 0.0
            patterns = [
                f"{err} ({count})" if len(err) != 0 else f'_ ({count})'
                for err, count in sorted(errors.items(), key=lambda x: x[1], reverse=True)
            ]
            error_pattern_list.append(
                [
                    word,
                    str(frequency),
                    str(total_errors),
                    f"{error_rate*100:.2f}",
                    ", ".join(patterns),
                ]
            )
    
        error_pattern_list.sort(key=lambda x: float(x[2]), reverse=True)
        error_pattern_list.sort(key=lambda x: float(x[3]), reverse=True)
        error_pattern_title = ['Entity', 'Counts', 'ErrorCounts', 'ErrorRate(%)', 'ErrorPatterns']
        output_path = os.path.join(output_dir, "error_patterns.tsv")
        write_file(output_path, [error_pattern_title] + error_pattern_list, sp="\t")


def main(
    rareword_list_path,
    reference_path,
    hypothesis_path,
    output_dir
):

    # Read rare words
    rare_words = [line[0] for line in read_file(rareword_list_path, sp=" ")]
    word2idx   = {word.replace(' ', ''):i for i, word in enumerate(rare_words)}

    # Sort by length
    rare_words = sorted(rare_words, key=lambda s: len(s), reverse=True)
    jieba.load_userdict(rareword_list_path)

    # Read reference and hypothesis files
    references = [[line[0], line[1:]] for line in read_file(reference_path, sp=" ")]
    hypotheses = [
        [line[0], [word for word in line[1:] if word]]
        for line in read_file(hypothesis_path, sp=" ")
    ]
    uttids     = [line[0] for line in references]

    evaluator = ASREvaluator(rare_words)

    # Process each reference-hypothesis pair
    for ref, hyp in tqdm(zip(references, hypotheses)):
        evaluator.process_utterance(ref, hyp)
    
    # Compute metrics and save results
    evaluator.compute_metrics()
    
    if output_dir is None:
        dump_dir = "/".join(hypothesis_path.split('/')[:-1])
        output_dir = f'{dump_dir}/analysis'

    evaluator.save_results(uttids, word2idx, output_dir=output_dir)


if __name__ == "__main__":

    # Define command-line arguments
    parser = argparse.ArgumentParser(description='ASR Evaluator')

    parser.add_argument('--rareword_list_path', type=str, required=True,
                        help='Path to the rare words list')
    parser.add_argument('--reference_path', type=str, required=True,
                        help='Path to the reference transcripts')
    parser.add_argument('--hypothesis_path', type=str, required=True,
                        help='Path to the hypothesis transcripts')
    parser.add_argument('--output_dir', type=str,
                        help='Directory to save the output results')

    args = parser.parse_args()

    rareword_list_path = args.rareword_list_path
    reference_path = args.reference_path
    hypothesis_path = args.hypothesis_path
    output_dir = args.output_dir
    main(
        rareword_list_path,
        reference_path,
        hypothesis_path,
        output_dir
    )

"""
python3 -m pyscripts.contextual.error_analysis.caluate_rareword_mer \
    --rareword_list_path "./local/contextual/rarewords/esun_earningcall.entity.txt" \
    --reference_path "./dump/raw/test/text" \
    --hypothesis_path "./exp/asr_whisper/run_medium_contextual_adapter_decoder_alpha0.9/decode_asr_whisper_contextual_adapter_decoder_c20_entity_earningcall_trie_asr_model_valid.loss.ave_10best_fixed/test/text"
"""