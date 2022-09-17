import os

utilities = [
    #"comet"
    #"unigram-f1"
    "chrf"
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
        command = "python -m scripts.models.{}.generate_predictions --utility={} --model-path=./saved_models/{}/{}/best/".format(model, util, util, model)
        os.system(command)


