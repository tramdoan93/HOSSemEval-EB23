# HOSSemEval-EB23: A Robust Dataset for Aspect Based Sentiment Analysis of Hospitality Reviews

This repository includes the annotated data and code implementations of the approaches used in our paper

HOSSemEva-EB23: A Robust Dataset for Asepect Based Sentiment Analysis of Hospitality Reviews in Multimedia Tools and Applications 2024.
## Short Summary

We offer a robust dataset of hostel reviews collected from Booking.com spanning 2020 to 2023. This dataset is designed to address the task of aspect-based sentiment analysis, where we aim to predict all sentiment quads (`aspect category`, `aspect term`, `sentiment polarity`)

## Data 
We release a new dataset, namely hos_eb23 under the data dir.
The dataset used in this study contains 7,012 records across six aspects: Amenity, Branding, Experience, Facility, Loyalty, and Service.

## Modeling

Our study introduces three models:

- TAS Models: Developed based on the approach by ([Wang et al.,2020](https://knowledge-representation.org/j.z.pan/pub/WYDL+2020.pdf))
- GAS: Adapted from the study by Zhang et al. (2021).
- ASQP: Based on the work by Zhang et al. (2021) Link to paper.

### 1. TAS Models ([Wang et al.,2020](https://knowledge-representation.org/j.z.pan/pub/WYDL+2020.pdf))
#### Reqirements

- pytorch: 1.0.1
- python: 3.6.8
- tensorflow: 1.13.1 ( only for creating BERT-Pytorch-model)
- pytorch-crf: 0.7.2
- numpy: 1.16.4
- nltk: 3.4.4
- sklearn: 0.21.2

#### 1.1 Data preprocessing

```commandline
$ cd your_workpath/TAS/data
$ python data_preprocessing_for_TAS.py --dataset absa_dataset
```
The preprocessed data is located in the folders absa_dataset/three_joint/BIO or absa_dataset/three_joint/TO. These folders correspond to the two tagging schemes, BIO and TO, as mentioned in the TAS-BERT paper by Wang et al.  ([Wang et al.,2020](https://knowledge-representation.org/j.z.pan/pub/WYDL+2020.pdf))
The structure of the preprocessed data is as follows: 

| Columns   | Description                                                           |
|-----------|-----------------------------------------------------------------------|
| sentence_id  | Id of the sentence                                                    |
| yes_no  | whether the sentence has corresponding sentiment in the corresponding aspect. The corresponding sentiment and aspect are given in aspect_sentiment.                                                              |
| aspect_sentiment  | <aspect, sentiment> pair of this line, such as "experience positive". |
| sentence  | Content of the sentence. |
| ner_tags  | label sequence for targets that have corresponding sentiment in the corresponding aspect. The corresponding sentiment and aspect are given in aspect_sentiment. |

**Preparing data for training**:
Download uncased [BERT-Based model](https://github.com/google-research/bert), unpack, place the folder in the root directory and run convert_tf_checkpoint_to_pytorch.py to create BERT-pytorch-model and run following commands to get preprocessed data:

#### 1.2 TAS Models
Our study includes two approaches:

- TAS-BERT: This approach is based on the methodology outlined in the research by ([Wang et al.,2020](https://knowledge-representation.org/j.z.pan/pub/WYDL+2020.pdf)).
- TAS-Transformers: We extend TAS-BERT by applying a variety of Transformer models to enhance the approach.
##### 1.2.1 TAS-BERT
Download [uncased BERT-Based model](https://github.com/google-research/bert), unpack and place the folder in the root directory and run `convert_tf_checkpoint_to_pytorch.py` to created BERT-pytorch-model.
###### Training & Testing
```bash
$ cd your_workpath/TAS/TAS-BERT/
$ python convert_tf_checkpoint_to_pytorch.py
```
Train and test a joint detection model:

```commandline
CUDA_VISIBLE_DEVICES=0 
python TAS_BERT_joint.py \
--data_dir  /your_workpath/TAS/data/absa_dataset/three_joint/BIO/ \
--output_dir /your_workpath/TAS/TAS-BERT/results/absa_dataset/three_joint/BIO/my_result \
--vocab_file uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin \
--tokenize_method word_split \
--use_crf \
--eval_test \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 24 \
--eval_batch_size 8 \
--learning_rate 2e-5 \
--num_train_epochs 30.0
```
Based on the BERT model you select, please specify the appropriate files for vocab_file, bert_config_file, and init_checkpoint.
The test results for each epoch will be stored in test_eps_*.txt in the output folder.

##### Evaluation
We selected the epoch that performed best `test_ep_{best_epoch}` on the TASD task and then evaluated its results across all subtasks.

- In order to evaluate on the TASD task, ASD task, TSD task ignoring implicit targets and TSD task considering implicit targets:
```commandline
python evaluation_for_TSD_ASD_TASD.py \
--output_dir /your_workpath/TAS/TAS-BERT/results/absa_dataset/three_joint/BIO/my_result \
--num_epochs 30 \
--tag_schema BIO
```

- Evaluation on the AD and SD task:

```bash
$ python evaluation_for_AD_SD.py \
--output_dir /your_workpath/TAS/TAS-BERT/results/absa_dataset/three_joint/BIO/my_result \
--num_epochs 30\
--tag_schema BIO\
--best_epoch_file_ad_sd 'test_ep_{best_epoch}'
```

##### 1.2.2 TAS-Transformers

We used a variety of Transformer models from the Huggingface library for the TAS-Transformers models.

###### Training & Testing
```bash
$ cd your_workpath/TAS/TAS-Transformers/
```
Train and test a joint detection model:

```commandline
CUDA_VISIBLE_DEVICES=0 
python TAS_BERT_joint.py \
--data_dir /your_workpath/TAS/data/absa_dataset/three_joint/BIO/ \
--output_dir /your_workpath/TAS/TAS-Transformers/results/absa_dataset/three_joint/BIO/my_result \
--model_name 'YituTech/conv-bert-base' \
--vocab_file /your_workpath/TAS/TAS-Transformers/Vocab/vocab.txt \
--tokenize_method word_split \
--use_crf \
--eval_test \
--do_lower_case \
--max_seq_length 64 \
--train_batch_size 32 \
--eval_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 30
```
The test results for each epoch will be stored in test_eps_*.txt in the output folder.

##### Evaluation
We selected the epoch that performed best  `test_ep_{best_epoch}` on the TASD task and then evaluated its results across all subtasks.
- In order to evaluate on the TASD task, ASD task, TSD task ignoring implicit targets and TSD task considering implicit targets:
```commandline
python evaluation_for_TSD_ASD_TASD.py \
--output_dir /your_workpath/TAS/TAS-Transformers/results/absa_dataset/three_joint/BIO/my_result \
--num_epochs 30 \
--tag_schema BIO
```

- Evaluation on the AD and SD task:

```bash
$ python evaluation_for_AD_SD.py \
--output_dir /your_workpath/TAS/TAS-Transformers/results/absa_dataset/three_joint/BIO/my_result \
--num_epochs 30\
--tag_schema BIO\
--best_epoch_file_ad_sd 'test_ep_{best_epoch}'
```

### 2. GAS ([Zhang et al.,2021](https://aclanthology.org/2021.acl-short.64.pdf))

#### Reqirements

- editdistance : 0.8.1
- pytorch_lightning : 0.8.1
- sentence_transformers : 3.0.1
- tf_keras : 2.15.0
- sentencepiece : 0.1.91


#### Training
To retrain the model with our data or train the model with your data annotated in our format, follow the syntax below:
``` 
cd /your_workpath/HOSSemp-EB23/GAS/
```
```bash
python main.py --task $task \
            --dataset $dataset \
            --model_name_or_path t5-base \
            --paradigm extraction \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 2 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20
```

- $task refers to one of the ABSA task in [`aope`,`uabsa`,`aste`,`tasd`]. In our study, we run experiment on `tasd`.
- $dataset: hos_eb23 is used in our study.

More details can be found in the `main.py`.
### 3. ASQP  ([Zhang et al., 2021](https://arxiv.org/pdf/2110.00796.pdf))
#### Reqirements

- editdistance : 0.8.1
- pytorch_lightning : 0.8.1
- sentence_transformers : 3.0.1
- tf_keras : 2.15.0
- sentencepiece : 0.1.91

#### Training

To retrain the model with our data or train the model with your data annotated in our format, follow the syntax below:

``` 
cd /your_workpath/HOSSemp-EB23/ASQP/
```
```bash
!python main.py --task $task \
            --dataset $dataset \
            --model_name_or_path t5-base \
            --n_gpu 0 \
            --do_train \
            --do_direct_eval \
            --train_batch_size 16 \
            --gradient_accumulation_steps 1 \
            --eval_batch_size 16 \
            --learning_rate 3e-4 \
            --num_train_epochs 20
```
- $task refers to one of the ABSA task in [`aope`,`uabsa`,`aste`,`tasd`]. In our study, we run experiment on `tasd`.
- $dataset: hos_eb23 is used in our study.

## Citation
If the code is used in your research, please star our repo and cite our paper as follows:
```
@article{doan2024hossemeval,
  title={HOSSemEval-EB23: a robust dataset for aspect-based sentiment analysis of hospitality reviews},
  author={Doan, Tram T and Tran, Thuan Q and Le, Dat T and Tran, Anh H and Nguyen, An T and Le, Tran Hoai An and Doan, Tran Nguyen Tung and Huynh, Son T and Nguyen, Binh T},
  journal={Multimedia Tools and Applications},
  pages={1--31},
  year={2024},
  publisher={Springer}
}
```
