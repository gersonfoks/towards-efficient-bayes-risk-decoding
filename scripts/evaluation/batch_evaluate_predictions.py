import os

utilities = [
    #"comet"
    #"unigram-f1",
    "chrf",
]


models = [
    "basic_model",
    "last_hidden_state_model",
    "full_dec_model",
    "full_dec_no_stat_model",
    "token_statistics_model"
]

for util in utilities:
    for model in models:
        command = "python -m scripts.evaluation.evaluate_predictions --predictions-ref=./model_predictions/{}/{}_predictions.parquet --model-name={} --utility={}".format(util, model, model, util)
        os.system(command)


