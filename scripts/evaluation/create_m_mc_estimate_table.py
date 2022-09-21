import json

utilities = [

    "unigram-f1",
    "chrf",
    "comet",
]




cols = [
    [
        1,2,3,5,10,25,50, 100
    ]
]

for util in utilities:
    ref = './results/{}/m_mc_estimates.json'.format(util)
    with open(ref, 'r') as f:
        temp = json.load(f)
        cols.append(temp)


def cols_to_table(cols):
    table = ''
    for i in range(len(cols[0])):

        for j, c in enumerate(cols):
            if j == 0:
                table += '{}'.format(c[i])
            else:
                table += ' & {:.1e}'.format(c[i])
        table += '\\\\ \n'
    print(table)
cols_to_table(cols)
