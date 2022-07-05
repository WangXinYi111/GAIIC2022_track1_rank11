#! /bin/bash
python code/make_neg_sample_v1.py
python code/code_textcnn/train.py
python code/code_Bert/train_Bert1.py
python code/make_neg_sample_v2.py
python code/code_Bert/train_Bert2.py