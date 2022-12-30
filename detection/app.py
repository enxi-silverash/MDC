import os

print("app.py", os.getcwd())

from detection.text2phoneme import *
from detection.test_detection import *
from detection.audio_prepare import *
import numpy as np


class application:
    phonemes = []  # 音素
    words = []  # 文本
    phoneme_str = ""  # 音素字符串，以下的函数都是处理的音素字符串
    words_phonemes_match = []  # 文本-音素匹配列表
    ark_file = './detection/fbank.ark'
    scp_file = './detection/fbank.scp'

    # l1_path = 'l1_path'
    # l2_path = 'l2_path'

    # audio_path=""      #也许以后需要

    def get_all(self, text):
        self.words, self.phonemes, self.phoneme_str, self.words_phonemes_match = text2phoneme_interface(text)
        # print("这里这里words:",self.words)
        # print("这里这里phonemes:",self.phonemes)
        # print("这里这里words_phonemes_match:",self.words_phonemes_match)

    def words_detection_result(self, phoneme_detection_result, normal_text):
        '''
        用于获得文本单词级别的检测结果
        输入:
            phoneme_detection_result:[0,1,0,1]
            setence1:[[word1,phoneme1,phoneme2],[word2,phoneme1,phoneme2]...]
        输出:
            words_detection_res:[0,1,0,1]
        '''
        index = 0
        words_detection_res = []
        k = 0
        res_score = []
        for word in normal_text:
            k = k + 1
            length = len(word)
            res = 0
            dete = 0
            for i in range(length):
                # print(i)
                if (index + i >= len(phoneme_detection_result)):
                    break
                res += phoneme_detection_result[index + i]

            # if (res / length > 0.2):
            #     words_detection_res.append(0)
            # else:
            #     words_detection_res.append(1)

            # if (k <= 1 or k == len(normal_text) or (len(word) <= 3 and res / length > 0.4) or (
            # len(word) > 3 and res / length > 0.1) or (len(word) < 3 and res / length == 0)):
            #     if ((word == 'and' or word == 'the' or word == 'for') and (res / length < 0.35) and len(
            #             normal_text[k - 2]) > 3):
            #         words_detection_res[-1] = 1
            #     words_detection_res.append(0)
            # else:
            #     words_detection_res.append(1)

            done = ['and', 'the', 'for']
            res_score.append({word: res / length})

            if (k <= 1 or k == len(normal_text) or (len(word) > 3 and res / length > 0.2)
                    or (len(word) < 3) or (word == 'and' or word == 'the' or word == 'for')):
                if (len(word) <= 3 and (word == 'and' or word == 'the' or word == 'for')):
                    if (k >= 1 and res / length < 0.35 and len(normal_text[k - 1]) / 3 >= 2 and
                            done.count(normal_text[k - 1]) == 0):
                        print(done.count(normal_text[k - 1]) == 0)
                        words_detection_res[-1] = 1
                        print("1111", normal_text[k - 1], res / length, 1)
                    words_detection_res.append(0)
                elif (len(word) < 3 and res / length == 0):
                    words_detection_res.append(0)
                else:
                    words_detection_res.append(0)
            else:
                print(word, res / length, 0)
                words_detection_res.append(1)

            index = index + length

        print("words_detection_res", words_detection_res)
        return words_detection_res, res_score

    def data_pre_process(self, audio_path):
        # make_spectrum(dataset_path, ark_file, scp_file)
        make_spectrum_test(audio_path, self.ark_file, self.scp_file)

    def dynamic_edit(self, hypothesis, reference):
        """编辑距离
        计算两个序列的levenshtein distance，可用于计算 WER/CER
        输入单词则检测单词，输入句子则检测句子，对于我而言其实要输入的就是直接的单词匹配？
        C: correct
        W: wrong
        I: insert
        D: delete
        S: substitution

        :param hypothesis: 预测序列
        :param reference: 真实序列
        :return: 1: 错误操作，所需要的 S，D，I 操作的次数;
                2: ref 与 hyp 的所有对齐下标
                3: 返回 C、W、S、D、I 各自的数量
        """
        print("dynamic_edit function used")
        print("hypothsis:", hypothesis)
        print("reference:", reference)
        len_hyp = len(hypothesis)
        len_ref = len(reference)
        cost_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int16)

        # 记录所有的操作，0-equal；1-insertion；2-deletion；3-substitution
        ops_matrix = np.zeros((len_hyp + 1, len_ref + 1), dtype=np.int8)

        for i in range(len_hyp + 1):
            cost_matrix[i][0] = i
        for j in range(len_ref + 1):
            cost_matrix[0][j] = j

        # 生成 cost 矩阵和 operation矩阵，i:外层hyp，j:内层ref
        for i in range(1, len_hyp + 1):
            for j in range(1, len_ref + 1):
                if hypothesis[i - 1] == reference[j - 1]:
                    cost_matrix[i][j] = cost_matrix[i - 1][j - 1]
                else:
                    substitution = cost_matrix[i - 1][j - 1] + 1
                    insertion = cost_matrix[i - 1][j] + 1
                    deletion = cost_matrix[i][j - 1] + 1

                    # compare_val = [insertion, deletion, substitution]   # 优先级
                    compare_val = [substitution, insertion, deletion]  # 优先级

                    min_val = min(compare_val)
                    operation_idx = compare_val.index(min_val) + 1
                    cost_matrix[i][j] = min_val
                    ops_matrix[i][j] = operation_idx

        match_idx = []  # 保存 hyp与ref 中所有对齐的元素下标
        i = len_hyp
        j = len_ref
        nb_map = {"N": len_ref, "C": 0, "W": 0, "I": 0, "D": 0, "S": 0}
        while i >= 0 or j >= 0:
            i_idx = max(0, i)
            j_idx = max(0, j)

            if ops_matrix[i_idx][j_idx] == 0:  # correct
                if i - 1 >= 0 and j - 1 >= 0:
                    match_idx.append((j - 1, i - 1))
                    nb_map['C'] += 1

                # 出边界后，这里仍然使用，应为第一行与第一列必然是全零的
                i -= 1
                j -= 1
            # elif ops_matrix[i_idx][j_idx] == 1:   # insert
            elif ops_matrix[i_idx][j_idx] == 2:  # insert
                i -= 1
                nb_map['I'] += 1
            # elif ops_matrix[i_idx][j_idx] == 2:   # delete
            elif ops_matrix[i_idx][j_idx] == 3:  # delete
                j -= 1
                nb_map['D'] += 1
            # elif ops_matrix[i_idx][j_idx] == 3:   # substitute
            elif ops_matrix[i_idx][j_idx] == 1:  # substitute
                i -= 1
                j -= 1
                nb_map['S'] += 1

            # 出边界处理
            if i < 0 and j >= 0:
                nb_map['D'] += 1
            elif j < 0 and i >= 0:
                nb_map['I'] += 1

        match_idx.reverse()
        wrong_cnt = cost_matrix[len_hyp][len_ref]
        nb_map["W"] = wrong_cnt

        # print("ref: %s" % " ".join(reference))
        # print("hyp: %s" % " ".join(hypothesis))
        # print(nb_map)
        # print("match_idx: %s" % str(match_idx))
        # print("wrong_cnt:",wrong_cnt)
        # print("match_idx:",match_idx)
        # print("nb_map:",nb_map)
        return wrong_cnt, match_idx, nb_map
        # detection_result=wrong_cnt*100/nb_map["N"]

        # return detection_result

    def char_detection_result(self, normal_phonemes, detection_phonemes):
        '''
        用于把对齐后的模型检测的phoneme与标准进行比较
        基于比较结果,获得phoneme级别的检测结果
        输入:normal_phonemes,detection_phonemes
        输出:char_detection_res
        '''
        print("normal_phonemes:", normal_phonemes)
        print("detection_phonemes", detection_phonemes)
        # 这里的normal_phonemes是字符串
        char_detection_res = np.zeros(len(normal_phonemes))

        _, match_idx, _ = self.dynamic_edit(detection_phonemes, normal_phonemes)
        print("match_idx:", match_idx)
        for res in match_idx:
            # 我知道的是两个序列的配对信息，那么我需要的是元素之间的信息，所以我要
            # 拆分获得单独序列，然后对每个词循环，
            char_detection_res[res[0]] = 1  # 0为标准文本,首先匹配序列，然后根据index筛选词
        phoneme_detection_res = []
        print("normal---------------", normal_phonemes)
        index = 0
        temps = normal_phonemes.split()
        for temp in temps:
            length = len(temp)
            temp_word_res = 1
            for i in range(length):
                if (char_detection_res[index + i] != 1):
                    temp_word_res = 0
                    break
            phoneme_detection_res.append(temp_word_res)
            index = index + length + 1  # 注意空格
        # print(len(phome_detection_res),len(char_detection_res))

        return char_detection_res, phoneme_detection_res

    def load_detection_model(self):
        opts = lode_config(config_file='ctc_config.yaml')
        model, device = load_model(opts, model_path="./detection/ctc_best_model_old.pkl")
        return opts, model, device

    def detection_assembel(self, opts, model, device):
        '''
            目前需要的数据文件(地址在配置文件中配置)
            transcript_phn_text:标注音频中音素序列(后序大概率不需要)
            fbank.scp:mel频谱索引文件(可以看一下内容,这两个使用liberosa可以提到)
            fbank.ark:mel频谱存放文件
            lm_phone_bg.arpa:暂时不知
            wrd_text:文字序列
            phn_text:文字对应的音素序列
        '''
        # 配置载入,需要修改的就是文件路径
        # opts = lode_config(config_file='ctc_config.yaml')
        vocab_file = opts.vocab_file
        print("vocabb_file", vocab_file)
        vocab = Vocab(vocab_file)
        print("vocabbb:", vocab)
        beam_width = opts.beam_width
        decoder_type = opts.decode_type
        vocab_file = opts.vocab_file

        if decoder_type == 'Greedy':
            decoder = GreedyDecoder(vocab.index2word, space_idx=-1, blank_index=0)
        else:
            decoder = BeamDecoder(vocab.index2word, beam_width=beam_width, blank_index=0, space_idx=-1,
                                  lm_path=opts.lm_path, lm_alpha=opts.lm_alpha)
        print("如果出现config_file问题,需要修改config_file路径")
        # 模型载入
        # model, decoder, vocab, device = load_model(opts, model_path="ctc_best_model_attention_aug.pkl")
        # 数据载入
        test_loader = load_data(opts, vocab)
        print(vocab)
        # 检测结果
        decodes = detection(test_loader, model, decoder, device)
        print("模型检测结果为")
        for x in decodes:
            print(x)
        print("decodes[0][1]", decodes[0][1])
        return decodes

    def data_pro_process(self, detection_phonemes):
        words_phonemes_match, words, phonemes = self.words_phonemes_match, self.words, self.phoneme_str
        # 将检测结果形成字符串
        # detection_phonemes=detection_assembel()[0][1]
        # detection_phonemes=''.join(detection_phonemes[0][1]) 原来是这个效果
        # print("detection---------", detection_phonemes)
        # 将输入句子的音素转化为字符串，实际等同于join
        # sentence_phonemes=text2phoneme_interface()
        # print("detection----------", phonemes)
        # print(phonemes[1])
        # print(detection_phonemes)
        # 获得phoneme级别的检测结果
        _, phoneme_detection_res = self.char_detection_result(self.phoneme_str, detection_phonemes)

        res, res_score = self.words_detection_result(phoneme_detection_res, words)  # 获得word级别的检测结果

        print(len(phoneme_detection_res), len(phonemes.split()))
        print("模型检测输出", detection_phonemes)  # 0是错误
        print("原句音素", phonemes)
        print("音素诊断结果", phoneme_detection_res)
        print("单词--音素匹配表", words_phonemes_match)
        print("原句单词", words)
        print("单词诊断结果", res)
        print("单词诊断分数", res_score)
        return [words, res]

# if __name__ == "__main__":
#     text = "she had your dark suit in greasy wash water all year"
#     audio_path = './arctic_a0085.wav'


#     application = application()

#     #模型载入
#     opts,model,device=application.load_detection_model()
#     # 文本预处理
#     application.get_all(text)
#     # 音频预处理
#     application.data_pre_process(audio_path)
#     # 模型检测
#     detection_phonemes = application.detection_assembel(opts,model,device)[0][1]
#     # 数据后处理
#     application.data_pro_process(detection_phonemes)
