class TranscriptionService:
    def __init__(self, hf_token: str):
        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name="stt_ru_conformer_ctc_large"
        )

        self.diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
    
    def process_audio(self, audio_path: str):
        y, sr = librosa.load(audio_file, sr=16000, mono=True)
        audio_duration = len(y)/sr

        outputs = self.diarization_pipeline({
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
        transcription = self.asr_model.transcribe([audio_file])
        full_text_hypothesis = transcription[0]

        if hasattr(full_text_hypothesis, 'text'):
            predicted_text = full_text_hypothesis.text
        else:
            predicted_text = str(full_text_hypothesis)
        
        return {"result": predicted_text}