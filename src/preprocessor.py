import torchaudio
import torch


def preprocess_proto(audio_path: str):
    try:
        pcm, sr = torchaudio.load(audio_path)
        print(f"Shape: {pcm.shape}")
    except Exception as e:
        raise RuntimeError(f"error: {e}")


if __name__ == '__main__':
    preprocess_proto("dataset/cat/1-34094-A-5.wav")
