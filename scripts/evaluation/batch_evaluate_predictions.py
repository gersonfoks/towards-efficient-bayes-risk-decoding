import os

utilities = [
    "unigram-f1",
    "chrf",
    "comet",
]


# models = [
#     # "basic_model",
#     # "last_hidden_state_model",
#     # "full_dec_model",
#     # "full_dec_no_stat_model",
#     # "token_statistics_model"
#     'comet_model',
#     'full_dec_comet_model'
# ]
#
# # for util in utilities:
# #     for model in models:
# #         command = "python -m scripts.evaluation.evaluate_predictions --predictions-ref=./model_predictions/{}/{}_predictions.parquet --model-name={} --utility={}".format(util, model, model, util)
# #         os.system(command)
# #
#
# #evaluate basic ref model
for util in utilities:
    for m in [1, 2, 3, 4, 5, 10, 25, 50, 100]:
        model_name = 'basic_reference_model_{}'.format(m)
        command = "python -m scripts.evaluation.evaluate_predictions --predictions-ref=./model_predictions/{}/basic_reference_model_{}_predictions.parquet --model-name={} --utility={}".format(util, m, model_name, util)
        os.system(command)



# for util in utilities:
#     for m in [1,2, 3, 4, 5 ]:
#         model_name = 'unigram_count_model_{}'.format(m)
#         command = "python -m scripts.evaluation.evaluate_predictions --predictions-ref=./model_predictions/{}/unigram_count_model_{}_predictions.parquet --model-name={} --utility={}".format(util, m, model_name, util)
#         os.system(command)