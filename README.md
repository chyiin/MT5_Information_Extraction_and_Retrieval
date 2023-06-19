# MT5 Information Extraction and Retrieval

Extract cosmetic product name, brand, category, etc.

Python 3.6.9
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

# Tasks & Datasets
| Scripts | Datasets | Target |
|--------|----------|--------|
| run_product.sh | dataset/product/*.csv | 品名，品牌 |
| run_category.sh | dataset/category/*.csv | 種類：精華，化妝水，... |
| run_post.sh | dataset/post/*.csv | 心得文 |
| run_tag.sh | dataset/tag/*.csv | 功效：美白，保濕， ... |
| run_summ.sh | dataset/summ/*.csv | 功效摘要 |

# Training
sh run_product.sh ```--mode train```
```
python3 main.py --mode train \
--train_filename dataset/product/training_dataset.csv \
--validation_filename dataset/product/validation_dataset.csv \
--test_filename dataset/product/testing_dataset.csv \
--task product \
--batch 2 \
--epoch 5 \
--learning_rate 0.0001 \
--input_length 1024 \
--output_length 50 \
--train_size 0.9 \
--wandb True \
--seed 42 \
--gpu 1
```

# Inference
sh run_product.sh ```--mode predict```
```
python3 main.py --mode predict \
--train_filename dataset/product/training_dataset.csv \
--validation_filename dataset/product/validation_dataset.csv \
--test_filename dataset/product/testing_dataset.csv \
--task product \
--batch 2 \
--epoch 5 \
--learning_rate 0.0001 \
--input_length 1024 \
--output_length 50 \
--train_size 0.9 \
--wandb True \
--seed 42 \
--gpu 1
```