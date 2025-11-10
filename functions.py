import jiwer
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.core import Annotation, Segment
import pandas as pd
from datetime import datetime
import string

class TranscriptionMetrics:
    @staticmethod
    def calculate_wer_cer(reference_text, hypothesis_text):
        try:
            wer_score = jiwer.wer(reference_text, hypothesis_text)
            cer_score = jiwer.cer(reference_text, hypothesis_text)
            
            measures = jiwer.compute_measures(reference_text, hypothesis_text)
            
            return {
                'wer': wer_score,
                'cer': cer_score,
                'hits': measures['hits'],
                'substitutions': measures['substitutions'],
                'deletions': measures['deletions'],
                'insertions': measures['insertions']
            }
        except Exception as e:
            print(f"Error calculating transcription metrics: {e}")
            return {'wer': 1.0, 'cer': 1.0, 'hits': 0, 'substitutions': 0, 'deletions': 0, 'insertions': 0}

class DiarizationMetrics:
    @staticmethod
    def calculate_der(reference_annotation, hypothesis_annotation):
        try:
            metric = DiarizationErrorRate()
            der_score = metric(reference_annotation, hypothesis_annotation)
            detailed_metrics = metric(reference_annotation, hypothesis_annotation, detailed=True)
            
            return {
                'der': der_score,
                'confusion': detailed_metrics['confusion'],
                'false_alarm': detailed_metrics['false alarm'],
                'missed_detection': detailed_metrics['missed detection']
            }
        except Exception as e:
            print(f"Error calculating diarization metrics: {e}")
            return {'der': 1.0, 'confusion': 0, 'false_alarm': 0, 'missed_detection': 0}

def create_sample_ground_truth():  
    reference_diarization = Annotation()
    
    reference_diarization[Segment(0.0, 1)] = "SPEAKER_00"
    reference_diarization[Segment(1.5, 2.6)] = "SPEAKER_01"
    reference_diarization[Segment(3.2, 4.6)] = "SPEAKER_00"
    reference_diarization[Segment(5.0, 6.5)] = "SPEAKER_01"
    reference_diarization[Segment(7.3, 8.5)] = "SPEAKER_00"
    reference_diarization[Segment(9.0, 9.7)] = "SPEAKER_01"

    return reference_diarization

def normalize_text(text):
    text = text.strip()
    text = text.lower()

    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    text = ' '.join(text.split())
    return text

def print_transcription_summary(transcription_metrics):
    print(f"Word Error Rate (WER): {transcription_metrics['wer']:.4f}")
    print(f"Character Error Rate (CER): {transcription_metrics['cer']:.4f}")
    print(f"Correct words: {transcription_metrics['hits']}")
    print(f"Substitutions: {transcription_metrics['substitutions']}")
    print(f"Deletions: {transcription_metrics['deletions']}")
    print(f"Insertions: {transcription_metrics['insertions']}")

    if transcription_metrics['wer'] < 0.1:
        print("Transcription: Excellent quality")
    elif transcription_metrics['wer'] < 0.2:
        print("Transcription: Good quality")
    elif transcription_metrics['wer'] < 0.3:
        print("Transcription: Acceptable quality")
    else:
        print("Transcription: Low quality")

def print_diarization_summary(diarization_metrics):
    print(f"Diarization Error Rate (DER): {diarization_metrics['der']:.4f}")
    print(f"Speaker confusion: {diarization_metrics['confusion']:.4f}")
    print(f"False alarms: {diarization_metrics['false_alarm']:.4f}")
    print(f"Missed detections: {diarization_metrics['missed_detection']:.4f}")
        
    if diarization_metrics['der'] < 0.1:
        print("Diarization: Excellent quality")
    elif diarization_metrics['der'] < 0.2:
        print("Diarization: Good quality")
    else:
        print("Diarization: Needs improvement")

def read_info_from_file(filename):
    try:
        with open(filename, 'r') as file:
            info = file.read().strip()
        return info
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    except Exception as e:
        print(f"Error during reading the information from {filename}: {e}")
        return None