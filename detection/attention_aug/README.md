“A Full Text-Dependent End to End Mispronunciation Detection and Diagnosis with Easy Data Augment Techniques“
https://arxiv.org/pdf/2104.08428.pdf



source code: https://github.com/cageyoko/CTC-Attention-Mispronunciation



Datasets: TIMIT, L2-Arctic 

( note: we suggest you need to use the preprocessed dataset we provided,

because it is difficult to process the source dataset )

( we delete the audio whose suffix is ".WAV.wav"  in TIMIT and 

process the dataset the same sampling rate 16000Hz by SOX )



Usage: 
1. cd attention_aug   (note: make sure attention_aug is your root directory)
2. Just need to change your kaldi_path in path.sh and your data_path in run.sh
3. ./run.sh  to get the decode sequence (decode_seq)
4. mv decode_seq ./result/hyp
   ./mdd_result.sh


