


python -m scripts.models.common.time_model --model-name=token_statistics_model --model-path=./saved_models/comet/token_statistics_model/best/
python -m scripts.models.common.time_model --model-name=full_dec_model --model-path=./saved_models/comet/full_dec_model/best/

python -m scripts.models.common.time_model --model-name=last_hidden_state_model --model-path=./saved_models/comet/last_hidden_state_model/best/
python -m scripts.models.common.time_model --model-name=full_dec_no_stat_model --model-path=./saved_models/comet/full_dec_no_stat_model/best/


python -m scripts.models.basic_model.time_model

python -m scripts.models.comet_model.time_model
python -m scripts.models.full_dec_comet_model.time_model


python -m scripts.models.unigram_count_model.time_model --n-model-references=1
python -m scripts.models.unigram_count_model.time_model --n-model-references=2
python -m scripts.models.unigram_count_model.time_model --n-model-references=3
python -m scripts.models.unigram_count_model.time_model --n-model-references=4
python -m scripts.models.unigram_count_model.time_model --n-model-references=5

