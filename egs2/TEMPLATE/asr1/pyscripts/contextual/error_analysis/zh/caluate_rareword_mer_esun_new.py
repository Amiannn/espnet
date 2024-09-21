import os
from collections import defaultdict

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


def find_rare_words(sentence, rare_words):
    """
    Find rare words in a sentence.
    For English words, check for exact matches with spaces.
    For non-English words, check for substring matches without spaces.
    """
    found_words = []
    sentence_no_space = sentence.replace(" ", "")
    for word in rare_words:
        if is_ascii(word):
            if (
                f" {word} " in sentence
                or sentence.startswith(f"{word} ")
                or sentence.endswith(f" {word}")
                or sentence == word
            ):
                found_words.append(word)
        else:
            if word in sentence_no_space:
                found_words.append(word)
    return found_words


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
            " ".join(ref_rare_words) if ref_rare_words else "correct"
        )
        self.hyp_rareword_sentences.append(
            " ".join(hyp_rare_words) if hyp_rare_words else "correct"
        )
        self.ref_common_sentences.append(
            " ".join(ref_common_words) if ref_common_words else "correct"
        )
        self.hyp_common_sentences.append(
            " ".join(hyp_common_words) if hyp_common_words else "correct"
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
        self.rare_eng_wer = wer(self.ref_rare_english, self.hyp_rare_english)
        self.rare_non_eng_cer = cer(
            self.ref_rare_non_english, self.hyp_rare_non_english
        )

        # Display metrics
        print(f"Overall MER: {self.overall_mer * 100:.2f}%")
        print(f"Rare Words MER: {self.rareword_mer * 100:.2f}%")
        print(f"Common Words MER: {self.common_mer * 100:.2f}%")
        print(f"Rare English Words WER: {self.rare_eng_wer * 100:.2f}%")
        print(f"Rare Non-English Words CER: {self.rare_non_eng_cer * 100:.2f}%")

    def save_results(self, output_dir="./exp/test"):
        """Write processed sentences and error patterns to files."""
        os.makedirs(output_dir, exist_ok=True)

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
                f"{err} ({count})"
                for err, count in sorted(errors.items(), key=lambda x: x[1], reverse=True)
            ]
            error_pattern_list.append(
                [
                    word,
                    str(frequency),
                    f"{error_rate:.2f}",
                    str(total_errors),
                    ", ".join(patterns),
                ]
            )

        error_pattern_list.sort(key=lambda x: int(x[1]), reverse=True)
        output_path = os.path.join(output_dir, "error_patterns.tsv")
        write_file(output_path, error_pattern_list, sp="\t")


def main():
    # Define file paths
    rareword_list_path = "./local/contextual/rarewords/rareword_f10_test.txt"
    reference_path = "./data/test/text"
    hypothesis_path = (
        "../asr1/exp/asr_whisper_medium_finetune_lr1e-5_adamw_wd1e-2_3epochs/"
        "decode_asr_whisper_noctc_greedy_asr_model_valid.acc.ave_3best/test/text"
    )

    # Read rare words
    rare_words = [line[0] for line in read_file(rareword_list_path, sp=" ")]

    # Read reference and hypothesis files
    references = [[line[0], line[1:]] for line in read_file(reference_path, sp=" ")]
    hypotheses = [
        [line[0], [word for word in line[1:] if word]]
        for line in read_file(hypothesis_path, sp=" ")
    ]

    evaluator = ASREvaluator(rare_words)

    # Process each reference-hypothesis pair
    for ref, hyp in zip(references, hypotheses):
        evaluator.process_utterance(ref, hyp)

    # Compute metrics and save results
    evaluator.compute_metrics()
    evaluator.save_results()


if __name__ == "__main__":
    main()
