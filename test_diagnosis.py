from detection.app import *


text = "she had your dark suit in greasy wash water all year"
audio_path = './test_files/test2.wav'


application = application()

# 模型载入
opts, model, device = application.load_detection_model()

# 文本预处理
application.get_all(text)

# 音频预处理
application.data_pre_process(audio_path)

# 模型检测
detection_phonemes = application.detection_assembel(opts, model, device)[0][1]

# 数据后处理
application.data_pro_process(detection_phonemes)
