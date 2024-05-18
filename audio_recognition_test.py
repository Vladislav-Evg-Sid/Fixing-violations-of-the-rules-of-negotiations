import speech_recognition as sr
import os

def get_file_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

file = 'wav_files/30к_872 КВ - 02.05.2024 08_40_27.wav'

files = get_file_paths('wav_files')

r = sr.Recognizer()
for i in range(len(files)):
    with sr.AudioFile(files[i]) as source:
        audio = r.record(source)  # read the entire audio file

        print(r.recognize_tensorflow(audio))

# with sr.AudioFile(file) as source:
#     audio = r.record(source)  # read the entire audio file
#
#     print(r.recognize_google(audio))