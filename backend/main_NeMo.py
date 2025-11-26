import nemo.collections.asr as nemo_asr
from pyannote.audio import Pipeline
import torch
import librosa
from huggingface_hub import login
import os
from functions import (
    TranscriptionMetrics,
    DiarizationMetrics,
    create_sample_ground_truth,
    print_diarization_summary,
    print_transcription_summary,
    read_info_from_file,
    normalize_text
)

token = read_info_from_file("hugging_face_token.txt")
if token:
    login(token=token)
    print("Token successfully loaded from file")
else:
    print("Failed to load token")
    exit(1)

reference_text = read_info_from_file("reference_text.txt")

print("Models downloading...")
try:
    asr_model = nemo_asr.models.EncDecCTCModel.from_pretrained(model_name="stt_en_quartznet15x5")
    print("Transcription model loaded: stt_en_quartznet15x5")
except Exception as e:
    print(f"Error loading NeMo model: {e}")

try:
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    print("Diarization model loaded")
except Exception as e:
    print(f"Error downloading the diarization model!")

print("All models successfully loaded!")

audio_file = "Talking-About-the-Weather.mp3"

y, sr = librosa.load(audio_file, sr=16000, mono=True)
audio_duration = len(y)/sr

print(f"Audio length: {audio_duration:.1f} seconds")

reference_diarization = create_sample_ground_truth()

print("Performing diarization...")
outputs = diarization_pipeline({
    "waveform": torch.from_numpy(y).unsqueeze(0).float(),
    "sample_rate": sr
})

predicted_diarization = outputs.speaker_diarization

diarization_segments = []
speakers = set()

speakers = set()
for segment, track, speaker in predicted_diarization.itertracks(yield_label=True):
    speakers.add(speaker)
    print(f"{speaker}: {segment.start:.1f}s - {segment.end:.1f}s")

print(f"\nSpeakers detected: {len(speakers)}")

print("\nPerforming transcription...")
transcription = asr_model.transcribe([audio_file])
full_text_hypothesis = transcription[0]

if hasattr(full_text_hypothesis, 'text'):
    predicted_text = full_text_hypothesis.text
else:
    predicted_text = str(full_text_hypothesis)

print("Predicted text normalization...")
predicted_text = normalize_text(predicted_text)

print("Reference text normalization...")
reference_text = normalize_text(reference_text)

print("="*60)
print("== Transcription metrics ==")
print("===========================")
transcription_metrics = TranscriptionMetrics.calculate_wer_cer(reference_text, predicted_text)
print_transcription_summary(transcription_metrics)

print("="*60)
print("== Diarization metrics ==")
print("===========================")
diarization_metrics = DiarizationMetrics.calculate_der(predicted_diarization, reference_diarization)
print_diarization_summary(diarization_metrics)
