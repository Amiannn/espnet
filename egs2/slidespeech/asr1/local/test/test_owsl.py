# make sure espnet is installed: pip install espnet
from espnet2.bin.s2t_inference import Speech2Text

model = Speech2Text.from_pretrained(
  "espnet/owls_05B_180K"
)

speech, rate = soundfile.read("speech.wav")
speech = librosa.resample(speech, orig_sr=rate, target_sr=16000)
# make sure 16k sampling rate

text, *_ = model(speech)[0]
