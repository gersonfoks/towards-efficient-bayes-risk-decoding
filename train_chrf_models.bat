
python -m scripts.models.basic_model.train_model --config=./configs/predictive/chrf/basic_lstm.yml --utility=chrf --smoke-test

python -m scripts.models.full_dec_model.train_model --config=./configs/predictive/chrf/full_dec_model.yml --utility=chrf --smoke-test

python -m scripts.models.full_dec_model_no_stat.train_model --config=./configs/predictive/chrf/full_dec_model_no_stat.yml --utility=chrf --smoke-test


python -m scripts.models.last_hidden_state_model.train_model --config=./configs/predictive/chrf/last_hidden_state_model.yml --utility=chrf --smoke-test

python -m scripts.models.token_statistics_model.train_model --config=./configs/predictive/chrf/token_statistics_model.yml --utility=chrf --smoke-test
