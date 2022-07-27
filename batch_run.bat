

python -m scripts.train.train_model --utility=comet --config=./configs/token_statistics_attention_model.yml
python -m scripts.train.train_model --utility=comet --config=./configs/token_statistics_lstm_model.yml
python -m scripts.train.train_model --utility=comet --config=./configs/last_hidden_state_attention.yml
python -m scripts.train.train_model --utility=comet --config=./configs/last_hidden_state_lstm.yml




