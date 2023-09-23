import torchaudio

SAMPLE_WAV = "/share/nas165/amian/experiments/speech/espnet/egs2/aishell/asr1/dump/raw/org/dev/data/format.1/data_wav.ark"

metadata = torchaudio.info(SAMPLE_WAV)
print(metadata)