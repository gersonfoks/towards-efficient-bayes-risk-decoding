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

    # base_ref = './results/'
    # for m in ms:
    #     summary = {
    #
    #     }
    #     name = '{}_mc_estimate'.format(m)
    #     pretty_name = '{}-MC Estimate'.format(m)
    #     save_location = base_ref + name + '/'
    #     Path(save_location).mkdir(parents=True, exist_ok=True)
    #     df["predictions"] = df[name]
    #
    #     plot_rank_vs_predicted_util(df, save_location)
    #
    #     kendall_taus, lengths, one_hyp_count = plot_kendall_taus(df, save_location, pretty_name)
    #
    #     summary["mean_kendall_taus"] = np.mean(kendall_taus)
    #     summary["median_kendall_taus"] = np.median(kendall_taus)
    #     summary["std_kendall_taus"] = np.std(kendall_taus)
    #
    #     targets = df["target"].to_list()
    #     sources = df["source"].to_list()
    #
    #     best_predictions = get_highest_scoring_predictions(df)
    #     top_10_percent_prediction = get_top_p_predictions(df, 0.1)
    #
    #     top_10_comet_scores = wrapped_model.batch_predict(sources, top_10_percent_prediction, targets)
    #
    #     # sns.histplot(top_10_comet_scores)
    #
    #     summary["top_10_comet_mean"] = np.mean(top_10_comet_scores)
    #     summary["top_10_comet_median"] = np.median(top_10_comet_scores)
    #     summary["top_10_comet_std"] = np.std(top_10_comet_scores)
    #
    #     best_predictions_comet_scores = wrapped_model.batch_predict(sources, best_predictions, targets)
    #
    #     summary["best_comet_mean"] = np.mean(best_predictions_comet_scores)
    #     summary["best_comet_median"] = np.median(best_predictions_comet_scores)
    #     summary["best_comet_std"] = np.std(best_predictions_comet_scores)
    #
    #     # Save the summary
    #     summary_ref = save_location + 'summary.json'
    #
    #     with open(summary_ref, 'w') as fp:
    #         json.dump(summary, fp)

    # mse = [get_m_mc_error(df, m) for m in ms]
    #
    # kendall_tau = [get_kendall_tau(df, m) for m in ms]
    #
    #
    # summary = {
    #     "mse": mse
    #     "kendall_tau": kendall_tau,
    #
    # }
    #
    #
    #
    # save_location = './results/{}/m_mc_estimates.json'.format(args.utility)
    #
    # with open(save_location, "w") as f:
    #     json.dump(summary, f)


    # Next create kendall tau statistics

    result_table = ''

    # for m, r in zip(ms, results):
    #     result_table += '{} & {:.1e} \\\\\n'.format(m, r)
    # print(result_table)


if __name__ == '__main__':
    main()
