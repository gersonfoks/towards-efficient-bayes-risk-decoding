


python -m scripts.models.full_dec_model.train_model --config=./configs/predictive/unigramf1/full_dec_model.yml --utility=unigram-f1
python -m scripts.models.full_dec_no_stat_model.train_model --config=./configs/predictive/unigramf1/full_dec_no_stat_model.yml --utility=unigram-f1
python -m scripts.models.token_statistics_model.train_model --config=./configs/predictive/unigramf1/token_statistics_model.yml --utility=unigram-f1
python -m scripts.models.last_hidden_state_model.train_model --config=./configs/predictive/unigramf1/last_hidden_state_model.yml --utility=unigram-f1
python -m scripts.models.basic_model.train_model --config=./configs/predictive/unigramf1/basic_lstm.yml --utility=unigram-f1