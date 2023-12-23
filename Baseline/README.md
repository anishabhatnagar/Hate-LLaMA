# Reproducing HateMM Baseline Results

This README covers the steps to reproduce the baseline results from the [HateMM paper](https://arxiv.org/pdf/2305.03915.pdf) to compare against our proposed model.

Please note that only the best performing model results are being reproduced here. 

## Requirements 

Make sure to use **Python3** when running the scripts. The package requirements can be obtained by running `pip install -r requirements.txt`.

------------------------------------------
***Folder Description*** :point_left:
------------------------------------------
~~~

./Data --> Contains dataset split details

. --> Python scripts to extract features

./Embeddings --> Generated embeddings and utils  
./Dataset_Images --> Extracted video frames

Note:Ensure that the "Embeddings" and "Dataset_Images" folders are created to store generated embeddings and other utility data.
~~~

------------------------------------------
***Dataset***
------------------------------------------

[Download the HateMM dataset](https://zenodo.org/records/7799469)


After downloading, extract the audio from videos and place them in the following locations:

```plaintext
Embeddings/train_audios_new --> train audio files
Embeddings/val_audios_new --> val audio files
Embeddings/test_audios_new --> test audio files
```

------------------------------------------
***Implementation Steps***
------------------------------------------
# 1. Extract the Text-based Features 

Run BERTandHateXPlainEmbedding.py to generate BERT embeddings. The embeddings will be saved in the newly created "Embeddings/" folder.

# 2. Then, Extract the Audio Based Features

Run the following two files:

AudioMFCC_Feat_andSpectrumGen.py: Extracts MFCC features from the audio files (.wav) and saves the spectrums in the Audio Plots.

AudioVGG19andInceptionFeat.py: Extracts audio features using VGG19 and saves the embeddings in the "Embeddings/" folder.
  
# 3. Extract all the video frames.

Run frameExtract.py to extract all frames from videos and save them in the newly created "Dataset_Images" folder.

# 4. Extract the Video Based Features

Run the following two files:

AudioVGG19andInceptionFeat.py: Extracts video features using Inception from the frames saved in "Dataset_Images" and saves the embeddings in "Embeddings/".

Model-ViT_featureExtract.py: Extracts video features using ViT from the frames saved in "Dataset_Images" and saves the embeddings in "Embeddings/".

# 5 Finally, run the Multimodal Model

Run MultiModalFusionModelfoldWise.py to fuse all generated embeddings and perform hate content classification.

