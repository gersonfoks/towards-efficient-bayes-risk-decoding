import argparse

import numpy as np

from tqdm import tqdm

from utilities.evaluation import plot_rank_vs_predicted_util, plot_kendall_taus, get_highest_scoring_predictions, \
    get_top_p_predictions, evaluate_predictions
from utilities.misc import load_bayes_risk_dataframe, map_to_utility


class GetMcEstimate:

    def __init__(self, n_references):

        self.n_references = n_references

    def __call__(self, x):


        count = np.array(x["references_count"])
        mc_estimates = [

        ]
        for utilities in np.array(x["utilities"]):

            probs = count / np.sum(count)
            # From the list of references pick n random
            picked_utilities = np.random.choice(utilities, size=self.n_references, p=probs, replace=True).tolist()
            mc_estimates.append(np.mean(picked_utilities))
        return np.array(mc_estimates)


def add_m_mc_estimate(df, m):
    mc_estimate_map = GetMcEstimate(m)

    df["{}_mc_estimate".format(m)] = df.progress_apply(mc_estimate_map, axis=1)
    return df
def get_m_mc_error(df, m):
    def calc_differences(x):
        difference = (x["utility"] - np.array(x["{}_mc_estimate".format(m)]))
        return difference ** 2

    df["differences"] = df.progress_apply(calc_differences, axis=1)

    mse = df["differences"].mean()
    return mse





def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Calculates the m-MC estimate for a given utility')

    parser.add_argument('--seed', type=int, default=0,
                        help="seed number (when we need different samples, also used for identification)")

    parser.add_argument('--utility', type=str,
                        default='comet',
                        help='Utility function used')

    args = parser.parse_args()

    tqdm.pandas()

    df = load_bayes_risk_dataframe('ancestral', 100, 1000, 'test', args.utility, args.seed, )
    df["utility"] = df[["utilities", 'references_count']].apply(map_to_utility, axis=1)

    #df = df.explode(['hypotheses', 'utilities', 'utility'])

    ms = [
        1, 2,
        3, 4,
        5, 10, 25, 50, 100
    ]

    summary = {

    }


    for m in ms:
        df = add_m_mc_estimate(df, m)

        model_name = "{}_mc_estimate".format(m)
        pretty_name = "-MC Estimate".format(m)
        df["predictions"] = df[model_name]



        evaluate_predictions(df, model_name, pretty_name, args.utility)




if __name__ == '__main__':
    main()
