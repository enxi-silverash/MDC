import matplotlib.pyplot as plt
import librosa
import numpy as np
import struct
import scipy
import os
import librosa.display
import torchaudio
import kaldiio

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}
audio_conf = {"sample_rate": 16000, 'window_size': 0.025, 'window_stride': 0.01, 'window': 'hamming'}


# 将输出音频特征转化为二进制数据流的格式写入ark文件
class KaldiWriteOut(object):
    def __init__(self, ark_path, scp_path):
        self.ark_path = ark_path
        self.scp_path = scp_path
        self.ark_file_write = open(ark_path, 'wb')
        self.scp_file_write = open(scp_path, 'w')
        self.pos = 0

    def write_kaldi_mat(self, utt_id, utt_mat):
        utt_mat = np.ascontiguousarray(utt_mat, dtype=np.float32)
        rows, cols = utt_mat.shape
        # 将字符串压缩为二进制数据
        self.ark_file_write.write(struct.pack('<%ds' % (len(utt_id)), utt_id.encode('utf-8')))
        self.ark_file_write.write(struct.pack('<cxcccc',
                                              ' '.encode('utf-8'),
                                              'B'.encode('utf-8'), 'F'.encode('utf-8'), 'M'.encode('utf-8'),
                                              ' '.encode('utf-8')))
        self.ark_file_write.write(struct.pack('<bi', 4, rows))
        self.ark_file_write.write(struct.pack('<bi', 4, cols))
        self.ark_file_write.write(utt_mat.tobytes())
        self.pos += len(utt_id) + 1
        self.scp_file_write.write(utt_id + ' ' + self.ark_path + ':' + str(self.pos) + '\n')
        self.pos += 3 * 5 + (rows * cols * 4)

    def close(self):
        self.ark_file_write.close()
        self.scp_file_write.close()


# 读取音频文件的路径
# def read_path(dataset_path):
def read_path(l1_path, l2_path):
    # l1_path = 'l1_path'
    # l2_path = 'l2_path'
    audio_path = 'path.txt'
    # 读取L1音频文件的路径
    with open(l1_path, 'r') as rf:
        note = open(audio_path, mode='w')
        for lines in rf.readlines():
            segment = lines.split('\n')[0].split('.')[0].split('/')
            print(segment)
            note.write(segment[-2] + '_' + segment[-1] + ' ' + lines)  # \n 换行符
        note.close()

    # 读取L2音频文件的路径
    with open(l2_path, 'r') as rf:
        note = open(audio_path, mode='a')
        for lines in rf.readlines():
            segment = lines.split('\n')[0].split('.')[0].split('/')
            print(segment)
            note.write(segment[-3] + '_' + segment[-1] + ' ' + lines)  # \n 换行符
        note.close()

    return audio_path


# 导入音频文件
def load_audio(path):
    """
    Input:
        path     : string 载入音频的路径
    Output:
        sound    : numpaudio.ndarraaudio 单声道音频数据，如果是多声道进行平均
    """

    # sound, _ = torchaudio.load(path)
    # sound = sound.numpy()

    # 利用librosa库导入音频文件
    sound, _ = librosa.load(path, sr=audio_conf['sample_rate'])
    sound = np.expand_dims(sound, axis=0)
    print(sound.shape)
    # sound = (sound - sound.mean(axis=1))
    return sound


def parse_audio(path, audio_conf, windows, normalize=True, visualization=False):
    """
    Input:
        path       : string 导入音频的路径
        audio_conf : dcit 求频谱的音频参数
        windows    : dict 加窗类型
    Output:
        spect      : ndarraaudio  每帧的频谱
    """
    audio = load_audio(path)

    # 预加重 y(n) = x(n) - pre_emphasis * x(n-1)
    pre_emphasis = 0.97

    # 正常的预加重
    audio = np.append(audio[0][0], audio[0][1:] - pre_emphasis * audio[0][:-1])

    # Kaldi的预加重
    # audio = np.append(audio[0][0] - pre_emphasis * audio[0][0], audio[0][1:] - pre_emphasis * audio[0][:-1])

    # 音频标准化
    audio = (audio - audio.mean()) / audio.std()
    audio = np.expand_dims(audio, axis=0)
    print('audio', audio.shape)

    # 分帧加窗
    audio_length = audio.shape[1]
    # n_fft = 512
    # fft点数
    n_fft = int(audio_conf['sample_rate'] * audio_conf["window_size"])
    print('n_fft', n_fft)
    # 帧长
    win_length = n_fft
    # 帧移
    hop_length = int(audio_conf['sample_rate'] * audio_conf['window_stride'])
    print('hop_length', hop_length)
    # 帧数
    frame = audio_length // hop_length + 1
    print('frame', frame)
    # 窗函数
    window = windows[audio_conf['window']]

    # 利用短时傅里叶变换计算能量谱
    magnitude_spectrogram = np.abs(librosa.stft(audio,
                                                n_fft=n_fft,
                                                hop_length=hop_length,
                                                center=False,  # 以音频序列的开始作为傅里叶变换的中心
                                                win_length=win_length,
                                                window=window)) ** 2
    magnitude_spectrogram = magnitude_spectrogram.squeeze()
    print('magnitude_spectrogram', magnitude_spectrogram.shape)
    # print(magnitude_spectrogram)

    # 梅尔滤波器组
    mel_basis = librosa.filters.mel(sr=audio_conf['sample_rate'],
                                    fmin=20.0,
                                    fmax=7800.0,
                                    n_fft=n_fft,
                                    n_mels=80,
                                    # norm=None
                                    )
    print('mel_basis', mel_basis.shape)

    # 通过梅尔滤波器组的结果
    mel_spectrum = np.dot(mel_basis, magnitude_spectrogram)
    print('mel_spectrum', mel_spectrum.shape)

    # mel_spectrum = librosa.feature.melspectrogram(audio, sr=audio_conf['sample_rate'], n_fft=n_fft, hop_length=hop_length, n_mels=80,center=False).squeeze()
    # print('mel_spectrum', mel_spectrum.shape)

    # 能量谱
    logenergaudio = np.sum(magnitude_spectrogram, axis=0).reshape(1, mel_spectrum.shape[1])

    print('logenergaudio', logenergaudio.shape)
    # print(logenergaudio.mean())
    # print(logenergaudio.std())
    # logenergaudio = (logenergaudio - logenergaudio.mean()) / logenergaudio.std()
    # print(logenergaudio)

    # 取对数并将梅尔频谱和能量谱拼接
    spect = np.concatenate((librosa.power_to_db(mel_spectrum), np.log(logenergaudio)), axis=0)

    # 能量标准化
    if normalize:
        spect = (spect - spect.mean()) / spect.std()

    print('out_feature:', spect.transpose().shape)

    # 是否可视化梅尔频谱
    if visualization:
        librosa.display.specshow(spect.transpose(),
                                 hop_length=hop_length, n_fft=n_fft, win_length=win_length,
                                 sr=audio_conf['sample_rate'])
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.show()

    return spect.transpose()


# 生成梅尔频谱（训练阶段，此时输入是L1和L2混合数据集的音频路径文件）
# def make_spectrum(dataset_path, ark_file, scp_file):
def make_spectrum_train(l1_path, l2_path, ark_file, scp_file):
    # wave_path = read_path(dataset_path)
    # wave_path = dataset_path
    wave_path = read_path(l1_path, l2_path)
    arkwriter = KaldiWriteOut(ark_file, scp_file)
    with open(wave_path, 'r') as rf:
        i = 0
        for lines in rf.readlines():
            utt_id, path = lines.strip().split()
            utt_mat = parse_audio(path, audio_conf, windows, normalize=True)
            arkwriter.write_kaldi_mat(utt_id, utt_mat)
            print('\n')

            i += 1
            if i % 10 == 0:
                print("Processed %d sentences..." % i)
        arkwriter.close()
        print("Done. Processed %d sentences..." % i)


# 生成梅尔频谱（训练阶段，此时输入是一段音频文件）
def make_spectrum_test(audio_path, ark_file, scp_file):
    arkwriter = KaldiWriteOut(ark_file, scp_file)
    utt_mat = parse_audio(audio_path, audio_conf, windows, normalize=True)
    arkwriter.write_kaldi_mat('test', utt_mat)
    print("Processed %d sentences..." % 1)
    arkwriter.close()
    print("Done. Processed %d sentences..." % 1)

# if __name__ == '__main__':
#     # dataset_path = 'C:/Users/86156/Desktop/mfcc/TIMIT/data/TEST/DR4/FADG0'
#     # dataset_path = 'C:/Users/86156/Desktop/mfcc/l2'
#     l1_path = 'l1_path'
#     l2_path = 'l2_path'
#     ark_file = './fbank.ark'
#     scp_file = './fbank.scp'
#
#     windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
#                'bartlett': scipy.signal.bartlett}
#     audio_conf = {"sample_rate": 16000, 'window_size': 0.025, 'window_stride': 0.01, 'window': 'hamming'}
#
#     # make_spectrum(dataset_path, ark_file, scp_file)
#     make_spectrum_train(l1_path, l2_path, ark_file, scp_file)

# a = kaldiio.load_mat('./fbank.ark:10')
# librosa.display.specshow(a,
#                          hop_length=160, n_fft=400, win_length=400,
#                          sr=audio_conf['sample_rate'])
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel spectrogram')
# plt.tight_layout()
# plt.show()
# print(a)
