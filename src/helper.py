from pydub import AudioSegment
import librosa
import numpy as np
import pandas as pd

def convert_mp3_to_wav(music_file):
  sound = AudioSegment.from_mp3(music_file)
  return sound.export("music_file.wav",format="wav")

def extract_features(music_file):
  audio = convert_mp3_to_wav(music_file)

  y,sr = librosa.load("music_file.wav",duration=30)
  # y,sr = librosa.load("Data/genres_original/classical/classical.00051.wav",duration=30)

  values =[661794]

  chroma = librosa.feature.chroma_stft(y=y,sr=sr)
  values.append(np.mean(chroma))
  values.append(np.var(chroma))  

  rms = librosa.feature.rms(y=y)
  values.append(np.mean(rms))
  values.append(np.var(rms))

  spectral_c = librosa.feature.spectral_centroid(y=y,sr=sr)
  values.append(np.mean(spectral_c))
  values.append(np.var(spectral_c))

  spectral_b = librosa.feature.spectral_bandwidth(y=y,sr=sr)
  values.append(np.mean(spectral_b))
  values.append(np.var(spectral_b))

  rolloff = librosa.feature.spectral_rolloff(y=y,sr=sr)
  values.append(np.mean(rolloff))
  values.append(np.var(rolloff))

  zero_crossing = librosa.feature.zero_crossing_rate(y=y)
  values.append(np.mean(zero_crossing))
  values.append(np.var(zero_crossing))

  harmony = librosa.effects.harmonic(y=y)
  values.append(np.mean(harmony))
  values.append(np.var(harmony))

  tempo = librosa.beat.tempo(y=y, sr=sr)
  values.append(np.mean(tempo))

  mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, dct_type=2, norm='ortho', lifter=0)
  for i in mfcc:
    values.append(np.mean(i))
    values.append(np.var(i))
  
  values = np.array(values) 
  return values

def scaling_data(music_file_values):
    dff = pd.read_csv("dff.csv")
    dff = dff.iloc[:,1:]
    means = np.array([])
    standard_deviation = np.array([])
    scaled_data = np.array([])
    data = music_file_values

    i = 0
    for column in dff.head(0):
        if column!="filename" and column!="label":
            means = np.append(means,dff[column].mean())
            standard_deviation = np.append(standard_deviation,dff[column].std())
            scaled_data = np.append(scaled_data,(data[i]-means[i])/standard_deviation[i])
            i=i+1

    scaled_data = np.array([scaled_data])
    scaled_data[0][0] = 0.0
    return scaled_data

# Example use case
# audio_features = extract_features("metal.mp3")
# scaled_audio_features = scaling_data(audio_features)