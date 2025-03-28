from transformers import BertTokenizer, WhisperProcessor, WhisperForConditionalGeneration
import torch
import librosa

whisper = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

bert = BertTokenizer.from_pretrained("bert-base-uncased")

def transcribe(audio_file): # might need to replace with base open-ai whisper model to 1-line transcribe (built-in)
    audio = librosa.load(audio_file, sr=16000)
    input = whisper(audio, return_tensors="pt", sampling_rate = 16000)

    with torch.no_grad():
        predicted_ids = model.generate(input["input_values"])

    transcription = whisper.decode(predicted_ids[0])

    return transcription


def bert_tokenize(transcription):
    tokens = bert(transcription, return_tensors="pt", padding=True, truncation=True)
    return tokens


def process_audio(audio_file):
    transcription = transcribe(audio_file)
    tokens = bert_tokenize(transcription)
    return tokens