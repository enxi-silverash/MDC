from pathlib import Path
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

# load model manager
path = Path(__file__).parent / "TTS/.models.json"
manager = ModelManager(path)

# load models
model_path, config_path, model_item = manager.download_model('tts_models/multilingual/multi-dataset/your_tts')
synthesizer = Synthesizer(model_path,config_path)
i = 0

while True:
    text = input('The sentence you\'d like to clone: ')
    target = input('The target wav: ')
    # kick it
    wav = synthesizer.tts(
        text=text,
        speaker_wav=target,
        language_name="en"
    )

    # save the results
    print(" > Saving output ...")
    synthesizer.save_wav(wav, 'tts_output'+str(i)+'.wav')

    if input('Enter 1 to continue, else stop: ') != '1':
        break
    else:
        i += 1

