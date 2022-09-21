import ast

from utilities.constants import PRETTY_NAMES

utilities = [
    "unigram-f1",
    "chrf",
    "comet",
]


models = [
    'basic_lstm',
    'last_hidden_state_model',
    'token_statistics_model',
    'full_dec_model',
    'full_dec_no_stat_model',
    'comet_model',
    'full_dec_comet_model'
]

base_ref = './saved_models/'



all_results = []

for model in models:

    results = []
    for util in utilities:
        ref = base_ref + "{}/{}/overview.txt".format(util, model)

        with open(ref, 'r') as f:
            lines = f.readlines()[0]
            lines = lines.replace(" device='cuda:0'", '').replace("(tensor", '').replace(',),', ',')

            lines = ast.literal_eval(lines)
            results.append(lines[0][0])

    all_results.append(results)

def print_as_table(names, rows):
    table = ""
    for name, results in zip(names, rows):
        table += " {} ".format(name)
        for r in results:
            table += " & {:.1e}".format(r)
        table += " \\\\ \n "
    print(table)

names = [PRETTY_NAMES[model] for model in models]
print_as_table(names, all_results)


ms = [1,2,3,4,5,10, 25, 50 ,100]
basic_reference_models = [
    "basic_reference_model_{}".format(m) for m in ms
]

all_results = []
for model in basic_reference_models:

    results = []
    for util in utilities:
        ref = base_ref + "{}/{}/overview.txt".format(util, model)

        with open(ref, 'r') as f:
            lines = f.readlines()[0]
            lines = lines.replace(" device='cuda:0'", '').replace("(tensor", '').replace(',),', ',')

            lines = ast.literal_eval(lines)
            results.append(lines[0][0])

    all_results.append(results)
print_as_table(ms, all_results)



small_ms = [1,2,3,4,5,]
basic_reference_models = [
    "unigram_count_model_{}".format(m) for m in small_ms
]

all_results = []
for model in basic_reference_models:

    results = []
    for util in utilities:
        ref = base_ref + "{}/{}/overview.txt".format(util, model)

        with open(ref, 'r') as f:
            lines = f.readlines()[0]
            lines = lines.replace(" device='cuda:0'", '').replace("(tensor", '').replace(',),', ',')

            lines = ast.literal_eval(lines)
            results.append(lines[0][0])

    all_results.append(results)
print_as_table(small_ms, all_results)