# MT5 Information Extraction and Retrieval

Extract cosmetic product name, brand, category, etc.

Python 3.6.9
```
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Train
```sh run_product.sh```
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

Inference
```sh run_product.sh```
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