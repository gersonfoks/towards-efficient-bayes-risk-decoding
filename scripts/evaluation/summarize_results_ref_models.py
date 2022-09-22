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


ms = [1,2,3,4,5,10,25,50,100]




results = {

}

for utility in utilities:
    base = './results/{}/'.format(utility)

    for m in ms:
        summary_ref = base  + 'basic_reference_model_{}/summary.json'.format(m)
        with open(summary_ref, 'r') as f:
            summary = json.load(f)

            results[(utility, m)] = summary




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



create_table(ms, utilities, results, "MSE")

create_table(ms, utilities, results, "mean_kendall_taus")



def create_utility_table(models, utilities, results):

    columns = [
        models
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

create_utility_table(ms, utilities, results,)
