from pyannote.audio import Pipeline
from huggingface_hub import login
import librosa
import torch
import whisper
import warnings
from functions import (
    normalize_text,
    TranscriptionMetrics,
    DiarizationMetrics,
    create_sample_ground_truth,
    print_transcription_summary,
    print_diarization_summary,
    read_info_from_file
)

warnings.filterwarnings("ignore")

token = read_info_from_file("hugging_face_token.txt")
reference_text = read_info_from_file("reference_text.txt")
if token:
    print("Token was successfully downloaded from file!")
    login(token=token)
else:
    print("Error: token has not been downloaded!")
    exit(1)

print("Download diarization and transcription model...")
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
whisper_model = whisper.load_model("base")
print("Models are successfully downloaded!")

print("Load audio file")
audio_file = "Talking-About-the-Weather.mp3"
y, sr = librosa.load(audio_file, sr=16000, mono=True)
audio_duration = len(y)/sr
print(f"Audio length: {audio_duration:.1f} seconds")

print("Transcription...")
transcription_result = whisper_model.transcribe(audio_file)
predicted_text = transcription_result["text"]

print("Transcription text normalization...")
predicted_text = normalize_text(predicted_text)

print("Reference text normalization...")
reference_text = normalize_text(reference_text)

print(f"Transcription result:\n{predicted_text}")
print(f"Reference text:\n{reference_text}")

transcription_metrics = TranscriptionMetrics.calculate_wer_cer(reference_text, predicted_text)
print("="*60)
print("== Transcription summary ==")
print("====================")
print_transcription_summary(transcription_metrics)
print("="*60)

print("== Predicted data ==")
print("====================")
outputs = diarization_pipeline({
    "waveform": torch.from_numpy(y).unsqueeze(0).float(),
    "sample_rate": sr
})
reference_diarization = create_sample_ground_truth()
predicted_diarization = outputs.speaker_diarization

speakers = set()
for segment, track, speaker in predicted_diarization.itertracks(yield_label=True):
    speakers.add(speaker)
    print(f"{speaker}: {segment.start:.1f}s - {segment.end:.1f}s")

print("="*60)
print("== Diarization summary ==")
print("=========================")
diarization_metrics = DiarizationMetrics.calculate_der(reference_diarization, predicted_diarization)
print_diarization_summary(diarization_metrics)
print("="*60)
