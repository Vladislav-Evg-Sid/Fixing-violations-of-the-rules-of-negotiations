# from huggingsound import SpeechRecognitionModel
#
# model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-russian")
# audio_paths = ["test.wav"]
#
# transcriptions = model.transcribe(audio_paths)
# print(transcriptions)

from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
import torchaudio
import torch

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-large-ls960-ft")
model = HubertForSequenceClassification.from_pretrained("xbgoose/hubert-speech-emotion-recognition-russian-dusha-finetuned")
num2emotion = {0: 'neutral', 1: 'angry', 2: 'positive', 3: 'sad', 4: 'other'}

filepath = "test_3.wav"

waveform, sample_rate = torchaudio.load(filepath, normalize=True)
transform = torchaudio.transforms.Resample(sample_rate, 16000)
waveform = transform(waveform)

inputs = feature_extractor(
        waveform,
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
        max_length=16000 * 10,
        truncation=True
    )

logits = model(inputs['input_values'][0]).logits
predictions = torch.argmax(logits, dim=-1)
predicted_emotion = num2emotion[predictions.numpy()[0]]
print(predicted_emotion)