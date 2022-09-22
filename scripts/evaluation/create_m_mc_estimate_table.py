import json



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



ms = [
        1,2,3,5,10,25,50, 100
    ]
cols = [
    ms
]
results = {

}



for utility in utilities:


    for m in ms:
        base = './results/{}/{}_mc_estimate/'.format(utility.replace('_', '-'), m)
        summary_ref = base + 'summary.json'
        with open(summary_ref, 'r') as f:
            summary = json.load(f)

            results[(utility, m)] = summary
def create_table(utilities, results, statistic):

    columns = [
        ms
    ]

    for utility in utilities:
        new_col = []
        for m in ms:
            new_col.append(
                results[(utility, m)][statistic]
            )
        columns.append(new_col)
    print_as_table(columns)


# create_table(utilities, results, "MSE")
#
# create_table(utilities, results, "median_kendall_taus")
#

def create_utility_table(utilities, results):

    columns = [
        ms
    ]



    for utility in utilities:

        print(utility)
        new_col = []

        val_1 =  'best_{}_mean'.format(utility.replace('-', '_'))

        for m in ms:
            new_col.append(results[(utility, m)][val_1])
        columns.append(new_col)
        val_2 = 'top_10_{}_mean'.format(utility.replace('-', '_'))
        new_col = []
        for m in ms:
            new_col.append(results[(utility, m)][val_2])
        columns.append(new_col)

    print_as_table(columns)


create_utility_table(utilities, results)
