

python -m scripts.models.unigram_count_model.train_model --config=./configs/predictive/unigramf1/unigram_count_model.yml --n-model-references=10 --utility=unigram-f1
python -m scripts.models.unigram_count_model.train_model --config=./configs/predictive/unigramf1/unigram_count_model.yml --n-model-references=25 --utility=unigram-f1
python -m scripts.models.unigram_count_model.train_model --config=./configs/predictive/unigramf1/unigram_count_model.yml --n-model-references=50 --utility=unigram-f1
python -m scripts.models.unigram_count_model.train_model --config=./configs/predictive/unigramf1/unigram_count_model.yml --n-model-references=100 --utility=unigram-f1
