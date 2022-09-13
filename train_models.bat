


python -m scripts.models.basic_model.train_model --config=./configs/predictive/chrf/basic_model.yml --utility=chrf
python -m scripts.models.last_hidden_state_model.train_model --config=./configs/predictive/chrf/last_hidden_state_model.yml --utility=chrf
python -m scripts.models.token_statistics_model.train_model --config=./configs/predictive/chrf/token_statistics_model.yml --utility=chrf
python -m scripts.models.full_dec_model.train_model --config=./configs/predictive/chrf/full_dec_model.yml --utility=chrf
python -m scripts.models.full_dec_no_stat_model.train_model --config=./configs/predictive/chrf/full_dec_no_stat_model.yml --utility=chrf
python -m scripts.models.comet_model.train_model --config=./configs/predictive/chrf/comet_model.yml --utility=chrf





python -m scripts.models.comet_model.train_model --config=./configs/predictive/unigramf1/comet_model.yml --utility=unigram-f1

