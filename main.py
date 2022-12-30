# modules for GUI
import os
import librosa
from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import showwarning
import pyaudio
import wave
import pygame
import numpy as np
import datetime
from random import randint

# modules for TTS
from pathlib import Path
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
import shutil

# modules for detection
from detection.app import *


audio_file = ''
sentence = ''
audio_result = ''
diag_result = None
words_result = None


def select_fun():
    print('function select is running')
    with open('sentences.txt', 'r') as f:
        lines = f.readlines()
        print(lines)
    text = lines[randint(0, len(lines)-1)].strip('\n')
    print(text)
    process_sentence(text)


def input_fun():
    print('function input is running')
    input_window = Toplevel(root)
    input_window.title('input')
    input_window.geometry('400x220')
    blank1 = Label(input_window, height=1)
    blank1.pack()
    text = Label(input_window, text='Input the sentence you want to practice:', font=('Microsoft YaHei UI',12))
    text.pack(pady=15)
    entry = Entry(input_window, font=('Microsoft YaHei UI',12), relief='flat', width=40)
    entry.pack()

    container_in = Label(input_window)
    confirm_btn = Button(container_in, text="Confirm", command=lambda: confirm_fun(entry.get(), input_window), \
                        font=('Microsoft YaHei UI', 11), width=10, relief='groove')
    confirm_btn.pack(pady=5, padx=5, side=LEFT)
    cancel_btn = Button(container_in, text="Cancel", command=lambda: input_window.destroy(), \
                       font=('Microsoft YaHei UI', 11), width=10, relief='groove')
    cancel_btn.pack(pady=5, padx=5, side=RIGHT)
    container_in.pack(pady=20)


def confirm_fun(text, input_window):
    print('function confirm is running')
    if process_sentence(text):
        input_window.destroy()
    else:
        showwarning('warning', 'The sentence is either empty or too long!')


def process_sentence(text):
    global sentence
    text_cpy = text

    # the maximun length of a line is 40 letters
    if text == '':
        sentence = None
        return False

    words = text.split()
    heads = split_lines(words)

    if len(heads) == 0:
        text += '\n'
    elif len(heads) == 1:
        words[heads[0]] = '\n' + words[heads[0]]
        text = ' '.join(words)
    else:
        sentence = None
        return False

    sentence = text_cpy
    print('The sentence to practice is:', sentence)
    print('The aligned text is:\n', text)
    example_content.set(text)

    return True


def split_lines(words):
    # return the indexes of the head-words in each line
    heads = []
    i = 0
    length = 0
    while i < len(words):
        while length < 40 and i < len(words):
            length += len(words[i]) + 1
            i+=1
        if length >= 40 and i < len(words):
            heads.append(i)
            length = 0
    return heads


def record_fun():
    print('function record is running')
    CHUNK = 1024
    FORMAT = pyaudio.paInt16  # 16bit编码格式
    CHANNELS = 1  # 单声道
    RATE = 16000  # 16000采样频率
    rec_time = 60
    threshold = 7000

    out_file = 'temp/audio.wav'

    timer = 0

    p = pyaudio.PyAudio()
    # 创建音频流
    stream = p.open(format=FORMAT,  # 音频流wav格式
                    channels=CHANNELS,  # 单声道
                    rate=RATE,  # 采样率16000
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Start Recording...")

    frames = []  # 录制的音频流
    # 录制音频数据
    for i in range(0, int(RATE / CHUNK * rec_time)):
        data = stream.read(CHUNK)
        frames.append(data)
        audio_data = np.fromstring(data, dtype=np.short)
        temp = np.max(audio_data)
        if temp > threshold:
            timer = 0
        else:
            timer += 1
        if timer > 100:
            break

    # 录制完成
    stream.stop_stream()
    stream.close()
    p.terminate()

    print("Recording Done...")

    # 保存音频文件
    wf = wave.open(out_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    global audio_file
    audio_file = out_file


def upload_fun():
    print('function upload is running')
    global audio_file
    # source = filedialog.askopenfilename()
    # audio_file = source.split('/')[-1]
    # shutil.copyfile(source, audio_file)
    audio_file = filedialog.askopenfilename()
    if audio_file:
        print(audio_file)


def play_fun():
    print('function play_input is running')
    if audio_file:
        play_audio(audio_file)
    else:
        showwarning('warning', 'No audio file has been chosen!')


def process_fun():
    global audio_file, audio_result
    if not audio_file or not sentence:
        showwarning('warning', 'The sentence or audio is not chosen!')
        return None
    global audio_result, diag_result, words_result
    audio_result, dect_result = process_audio()

    # process words_result
    words, res = dect_result[0], dect_result[1]
    for i in range(len(res)):
        words[i] += (' ×' if res[i] else ' √') + (',' if i < len(res) - 1 else '.')
    words_result = words
    diag_result = res
    print('words_result: ', words_result)
    print('diag_result: ', diag_result)

    # save Practice log
    file_list = os.listdir('practice_log/')
    print('Files in practice_log:', file_list)
    times = int((len(file_list) - 1)/2 if len(file_list) > 1 else 0)
    with open('practice_log/log.txt', 'a') as f:
        # s = str(times) + ' -- ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +\
        #     ' -- ' + sentence + ' -- ' + words_result + '\n'
        s = str(times) + ' -- ' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + \
            ' -- ' + sentence + ' -- ' + ' '.join(words_result) + '\n'
        print('Pratice log:', s)
        f.write(s)

    file_path = os.path.join('practice_log/'+'audio_yours_'+str(times)+'.'+audio_file.split('.')[-1])
    result_path = os.path.join('practice_log/'+'audio_corrected_'+str(times)+'.'+audio_result.split('.')[-1])

    shutil.copyfile(audio_file, file_path)
    shutil.copyfile(audio_result, result_path)

    # removing the unnecessary files
    if audio_file == 'temp/audio.wav':
        os.unlink(audio_file)
        audio_file = file_path
    if audio_result == 'temp/tts_output.wav':
        os.unlink(audio_result)
        audio_result = result_path


    print("Practice log has been saved to /practice log/")


def process_audio():
    print("Internal function of processing audios")
    print('processing', audio_file)

    # Audio cloning
    wav = synthesizer.tts(text=sentence, speaker_wav=audio_file, language_name='en')
    # save the results
    print(" > Saving cloning output ...")
    synthesizer.save_wav(wav, 'temp/tts_output.wav')

    # preprocess text and audio
    global application

    application.get_all(remove_punc(sentence))
    application.data_pre_process(audio_file)
    detection_phonemes = application.detection_assembel(opts, model, device)[0][1]
    diag_result = application.data_pro_process(detection_phonemes)

    return 'temp/tts_output.wav', diag_result


def remove_punc(text):
    for i in text:
        if i == '.' or i == ',' or i == '!' or i == '?':
            text = text.replace(i, ' ')  # 将.删掉
        elif i == "'":
            text = text.replace(i, '')  # 将.删掉

    return text


def diagnosis_fun():
    print('function result is running')
    if not diag_result:
        showwarning('warning', 'The result is not yet generated!')
        return

    words = words_result
    heads = split_lines(words)

    result_text = ''
    print('The head words of each line is:', heads)
    if len(heads) == 0:
        result_text += '\n'
    elif len(heads) == 1:
        words[heads[0]] = '\n' + words[heads[0]]
        result_text = ' '.join(words)
    else:
        showwarning('warning', 'The result is too long to display!')

    print(result_text)
    result_content.set(result_text)
    score_points.set('Your score: '+str(int(diag_result.count(0)/len(diag_result)*100))+"/100")


def correction_fun():
    print('function play_correction is running')
    if audio_result:
        play_audio(audio_result)
    else:
        showwarning('warning', 'The cloned audio is not yet generated!')


def play_audio(audio):
    pygame.mixer.init()
    pygame.mixer.music.load(audio)
    pygame.mixer.music.play()
    time.sleep(librosa.get_duration(filename=audio))
    pygame.mixer.quit()


print('MDC initializing...')

# load TTS model manager
path = Path(__file__).parent / "TTS/.models.json"
manager = ModelManager(path)

# load TTS models
model_path, config_path, model_item = manager.download_model('tts_models/multilingual/multi-dataset/your_tts')
synthesizer = Synthesizer(model_path,config_path)

# load diagnosis models
application = application()
opts, model, device = application.load_detection_model()

root = Tk()
root.title('MDC')
root.geometry('400x650')

blank = Label(root, height=1)
blank.pack(pady=3)
title = Label(root, text="Mispronunciation\n Diagnosis and Correction",\
              width=20, height=4, font=('Microsoft YaHei UI',17))
title.pack(pady=4)

container1 = Label(root)
example_content = StringVar()
example_content.set("Please select a sentence to start your practice\n")
example = Label(container1, bg="#FFFFFF", textvariable=example_content, font=('Microsoft YaHei UI',12),\
                anchor='w', justify='left', relief='flat', width=36)
example.pack(pady=8)

result_content = StringVar()
result_content.set("Here is the result\n")
result = Label(container1, bg="#FFFFFF", textvariable=result_content, font=('Microsoft YaHei UI',12),\
                anchor='w', justify='left', relief='flat', width=36)
result.pack(pady=4)

container1.pack()

score_points = StringVar()
score_points.set('Your score: None/100')
container2 = Label(root, textvariable=score_points, font=('Microsoft YaHei UI',15))
container2.pack(pady=14)

container3 = Label(root)
container31 = Label(container3)
select_btn = Button(container31, text="Select", command=select_fun,\
                    font=('Microsoft YaHei UI',13), width=10, relief='groove')
select_btn.pack(padx=5, side=LEFT)
input_btn = Button(container31, text="Input", command=input_fun,\
                    font=('Microsoft YaHei UI',13), width=10, relief='groove')
input_btn.pack(padx=5, side=RIGHT)
container31.pack(pady=6)

container32 = Label(container3)
record_btn = Button(container32, text="Record", command=record_fun,\
                    font=('Microsoft YaHei UI',13), width=10, relief='groove')
record_btn.pack(padx=5, side=LEFT)
upload_btn = Button(container32, text="Upload", command=upload_fun,\
                    font=('Microsoft YaHei UI',13), width=10, relief='groove')
upload_btn.pack(padx=5, side=LEFT)
container32.pack(pady=6)

container33 = Label(container3)
play_btn = Button(container33, text="Play", command=play_fun,\
                    font=('Microsoft YaHei UI',13), width=10, relief='groove')
play_btn.pack(padx=5, side=LEFT)
process_btn = Button(container33, text="Process", command=process_fun,\
                    font=('Microsoft YaHei UI',13), width=10, relief='groove')
process_btn.pack(padx=5, side=LEFT)
container33.pack(pady=6)

container34 = Label(container3)
diagnosis_btn = Button(container34, text="Diagnosis", command=diagnosis_fun,\
                    font=('Microsoft YaHei UI',13), width=10, relief='groove')
diagnosis_btn.pack(padx=5, side=LEFT)
correction_btn = Button(container34, text="Correction", command=correction_fun,\
                    font=('Microsoft YaHei UI',13), width=10, relief='groove')
correction_btn.pack(padx=5, side=LEFT)
container34.pack(pady=6)

container3.pack(pady=5)


copyright = Label(root, text='copyright©2022.All Rights Reserved.',\
                  font=('Microsoft YaHei UI',8), fg="#666666")
copyright.pack(side=BOTTOM, pady=5)

root.mainloop()