import os

utilities = [
    "unigram-f1",
    "chrf",
    "comet",
]

models = [
    "basic_model",
    "last_hidden_state_model",
    "full_dec_model",
    "full_dec_no_stat_model",
    "token_statistics_model",
    'comet_model',
    'full_dec_comet_model',
]
#
for util in utilities:
    for model in models:
        command = "python -m scripts.models.{}.generate_predictions --utility={} --model-path=./saved_models/{}/{}/best/".format(model, util, util, model)
        os.system(command)

# Generate basic reference predictions

#
# for util in utilities:
#     for m in [1, 2,3,4,5,10,25,50, 100]:
#         model = 'basic_reference_model'
#         model_path = './saved_models/{}/{}/best/'.format(util, model + "_{}".format(m))
#         command = "python -m scripts.models.{}.generate_predictions --utility={} --model-path={} --n-references={}".format(
#             model, util, model_path, m)
#         os.system(command)

# for util in utilities:
#     for m in [1, 2, 3, 4, 5,]:
#         model = 'unigram_count_model'
#         model_path = './saved_models/{}/{}/best/'.format(util, model + "_{}".format(m))
#         command = "python -m scripts.models.{}.generate_predictions --utility={} --model-path={} --n-references={}".format(
#             model, util, model_path, m)
#         os.system(command)
