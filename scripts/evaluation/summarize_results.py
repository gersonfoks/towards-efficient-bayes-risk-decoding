import json

from utilities.constants import PRETTY_NAMES

def print_as_table(columns ):

    table = ''

    for i in range(len(columns[0])):
        for j, col in enumerate(columns):

            if j == 0:
                table += '{}'.format(col[i])
            else:
                table += " & {:.1e}".format(col[i])

        table += '\\\\ \n'

    print(table)

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


# statistics = [
#    'MSE',
#     'median_kendall_taus',
#     'mean_kendall_taus',
#     'top_10_{}_mean'.format(utility),
#     'best_comet_mean'.format(utility),
# ]


results = {

}

for utility in utilities:
    base = './results/{}/'.format(utility)

    for model in models:
        summary_ref = base + model + '/summary.json'
        with open(summary_ref, 'r') as f:
            summary = json.load(f)

            results[(utility, model)] = summary




# We print out the mean information


def create_table(models, utilities, results, statistic):

    columns = [models]

    for utility in utilities:
        new_col = []

        for model in models:

            new_col.append(
                results[(utility, model)][statistic]
            )
        columns.append(new_col)
    print_as_table(columns)



create_table(models, utilities, results, "MSE")

create_table(models, utilities, results, "median_kendall_taus")



def create_utility_table(models, utilities, results):
    names = [PRETTY_NAMES[name] for name in models]
    columns = [
        names
    ]



    for utility in utilities:

        print(utility)
        new_col = []

        val_1 =  'best_{}_mean'.format(utility.replace('-', '_'))

        for m in models:
            new_col.append(results[(utility, m)][val_1])
        columns.append(new_col)
        val_2 = 'top_10_{}_mean'.format(utility.replace('-', '_'))
        new_col = []
        for m in models:
            new_col.append(results[(utility, m)][val_2])
        columns.append(new_col)

    print_as_table(columns)

create_utility_table(models, utilities, results,)
#
#
#
#

#
# print("Kendall taus")
# print_as_table(results, models, ["median_kendall_taus", "mean_kendall_taus"])
#
# print("MSE")
# print_as_table(results, models, ["MSE"])
#
#
# print("Scores")
# print_as_table(results, models, ["best_{}_mean".format(utility), "top_10_{}_mean".format(utility), ])
#
#
#
#
#
