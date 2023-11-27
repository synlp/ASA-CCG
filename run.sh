
# train
python main.py --do_train --train_data_path=./data/train.tsv --dev_data_path=./data/dev.tsv --test_data_path=./data/test.tsv --use_bert --bert_model=/path/to/bert_base_cased --n_mlp=200 --max_seq_length=300 --train_batch_size=16 --eval_batch_size=16 --num_train_epochs=3 --warmup_proportion=0.1 --learning_rate=1e-5 --patient=10 --model_name=testing

