import librosa
from os import path
from pydub import AudioSegment

def convertor(src_name):
    # files                                                                         
    src = "Nirvana-Smells_Like_Teen_Spirit.mp3"
    dst = "test.wav"

    # convert wav to mp3                                                            
    sound = AudioSegment.from_mp3(src_name)
    sound.export(dst, format="wav")
    print("working")
    return True
