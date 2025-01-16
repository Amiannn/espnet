import os

def read_file(path, sp=' '):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip().split(sp) for line in f]

def write_file(path, data, sp=' '):
    with open(path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(sp.join(line) + '\n')

def merge_utterances(contexts, target_len):
    """
    Merges consecutive utterances to ensure each merged utterance has exactly target_len tokens.
    Assigns new unique IDs to merged utterances.

    Args:
        contexts (list of lists): Each sublist contains [ID, UTTERANCE].
        target_len (int): Desired number of tokens per merged utterance.

    Returns:
        list of lists: Processed contexts with merged utterances of uniform length.
    """
    merged_contexts = []
    buffer_tokens = []
    buffer_ids = []
    new_id_counter = 0  # Initialize a counter for new IDs

    for context in contexts:
        utt_id, utt = context
        tokens = utt.split()
        buffer_tokens.extend(tokens)
        buffer_ids.append(utt_id)

        # Merge as many utterances as possible from the buffer
        while len(buffer_tokens) >= target_len:
            # Take the first target_len tokens
            merged_tokens = buffer_tokens[:target_len]
            # Remove these tokens from the buffer
            buffer_tokens = buffer_tokens[target_len:]
            # Remove corresponding IDs
            buffer_ids = buffer_ids[target_len:]

            # Assign a new unique ID
            merged_id = f"merged_{new_id_counter}"
            new_id_counter += 1

            # Append the merged utterance
            merged_contexts.append([merged_id, ' '.join(merged_tokens)])

    # Optionally handle remaining tokens in the buffer
    # Here, we discard them. Uncomment below to include them as is.
    # if buffer_tokens:
    #     merged_id = f"merged_{new_id_counter}"
    #     merged_contexts.append([merged_id, ' '.join(buffer_tokens)])
    
    return merged_contexts

def data_augmentation(contexts, num):
    """
    Augments data by splitting long utterances into multiple shorter ones.

    Args:
        contexts (list of lists): Each sublist contains [ID, UTTERANCE].
        target_len (int): Desired number of tokens per merged utterance.

    Returns:
        list of lists: Processed contexts with augmented data.
    """
    augmented_contexts = []
    for n in range(num):
        for context in contexts:
            utt_id, utt = context
            new_id = f"{utt_id}_{n}"
            augmented_contexts.append([new_id, utt])
    return augmented_contexts

def analyze_lengths(contexts):
    lengths = [len(context[1].split()) for context in contexts]
    print(f"Max length: {max(lengths)}")
    print(f"Min length: {min(lengths)}")
    print(f"Average length: {sum(lengths)/len(lengths):.2f}")
    # Optionally, plot the distribution
    import matplotlib.pyplot as plt
    plt.hist(lengths, bins=50)
    plt.title("Utterance Length Distribution")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.show()

# Define paths
source_context_path = "./dump/raw/S95_sp/uttblist_f65536"
target_context_path = "./dump/raw/test/uttblist_f10"
dev_context_path    = "./dump/raw/dev/uttblist_f65536"

# Read contexts
source_contexts = [[d[0], " ".join(d[1:]).upper()] 
                   for d in read_file(source_context_path) if len(d[1:]) > 0]
target_contexts = [[d[0], " ".join(d[1:]).upper()] 
                   for d in read_file(target_context_path) if len(d[1:]) > 0]
dev_contexts    = [[d[0], " ".join(d[1:]).upper()] 
                   for d in read_file(dev_context_path) if len(d[1:]) > 0]

print(f"Source context sample: {source_contexts[:5]}")
print(f"Target context sample: {target_contexts[:5]}")
print(f"Dev context sample: {dev_contexts[:5]}")

# Optional: Analyze utterance lengths to choose TARGET_LEN
# analyze_lengths(source_contexts)
# analyze_lengths(target_contexts)
# analyze_lengths(dev_contexts)

# Define the target length
TARGET_LEN = 50  # Adjust this based on your dataset analysis

# Preprocess all contexts by merging utterances
source_contexts_merged = merge_utterances(source_contexts, TARGET_LEN)
target_contexts_merged = data_augmentation(merge_utterances(target_contexts, TARGET_LEN), num=3)
dev_contexts_merged    = data_augmentation(merge_utterances(dev_contexts, TARGET_LEN), num=3)

print(f"Source context merged sample: {source_contexts_merged[:5]}")
print(f"Target context merged sample: {target_contexts_merged[:5]}")
print(f"Dev context merged sample: {dev_contexts_merged[:5]}")

# Write the merged data to new files
output_source_path = "./dump/raw/S95_sp/uttblist_f65536_merged"
write_file(output_source_path, source_contexts_merged, sp=' ')

output_target_path = "./dump/raw/test/uttblist_f10_merged"
write_file(output_target_path, target_contexts_merged, sp=' ')

output_dev_path = "./dump/raw/dev/uttblist_f65536_merged"
write_file(output_dev_path, dev_contexts_merged, sp=' ')

print("Preprocessing complete. All utterances are now of uniform length through merging.")
