import json

from utilities.constants import PRETTY_NAMES

utility = 'comet'
base = './results/{}/'.format(utility)


models = [
    'basic_model',
    'last_hidden_state_model',
    'token_statistics_model',
    'full_dec_model',
]



statistics = [
   'MSE',
    'median_kendall_taus',
    'mean_kendall_taus',
    'top_10_comet_mean',
    'best_comet_mean',
]


results = {

}

for model in models:
    summary_ref = base + model + '/summary.json'
    with open(summary_ref, 'r') as f:
        summary = json.load(f)
        results[model] = summary





def print_as_table(results,models, statistics ):

    table_entries = ''

    for model in models:
        table_entries += PRETTY_NAMES[model] + ' & '
        for i, stat in enumerate(statistics):
            val = results[model][stat]

            table_entries += "{:.1e}".format(val)

            if i != len(statistics) - 1:
                table_entries += ' & '
        table_entries += '\\\\ \n'

    print(table_entries)

print_as_table(results, models, ["median_kendall_taus", "mean_kendall_taus"])

print_as_table(results, models, ["MSE"])

print_as_table(results, models, ["top_10_comet_mean", "best_comet_mean"])





