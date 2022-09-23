import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
from math import ceil
from tqdm import tqdm
from scipy.stats import kendalltau
import seaborn as sns
from comet import download_model, load_from_checkpoint

from utilities.Utility import NGramF, ChrF, BleurtUtility
from utilities.wrappers.CometWrapper import CometWrapper
from evaluate import load
def get_mse(val_df):
    temp = val_df["predictions"].to_list()
    all_predictions = []
    for t in tqdm(temp):
        all_predictions += t.tolist()

    all_predictions = np.array(all_predictions)
    temp = val_df["utility"].to_list()
    all_true_values = []
    for t in tqdm(temp):
        all_true_values += t
    all_true_values = np.array(all_true_values)
    MSE = np.mean((all_true_values - all_predictions)**2)
    return MSE

def plot_rank_vs_predicted_util(val_df, save_location, n_plots=5, seed=42):
    np.random.seed(seed)
    indices = np.random.choice(len(val_df.index), n_plots)
    for i in indices:
        fig = plt.figure()
        row = val_df.iloc[i]


        predicted_means = np.array(row["predictions"])
        target_means = row["utility"]
        source = row["source"]
        sorted_indices = np.argsort(target_means)[::-1]

        predicted_means_sorted = predicted_means[sorted_indices]
        x = np.arange(0, len(predicted_means_sorted))

        sorted_indices_predicted = np.argsort(predicted_means_sorted)[::-1]

        colors = ["b"] * len(predicted_means)

        top_10 = math.ceil(0.1 * len(sorted_indices_predicted))
        for j in range(top_10):
            c = sorted_indices_predicted[j]
            colors[c] = 'g'

        z = np.polyfit(x, predicted_means_sorted, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x), "r--")

        plt.scatter(x, predicted_means_sorted, marker='o', c=colors, )
        plt.xlabel('Rank')
        plt.ylabel("Predicted utility")
        plt.title("'{}'".format(source))

        plt.show()
        save_loc = save_location + 'rank_vs_predicted_{}.png'.format(i)
        fig.savefig(save_loc, dpi=fig.dpi)




def map_to_top_p(x, p):
    '''
    First filters the top p percentage
    Then keep the highest scoring one based on the ground truth
    '''

    if len(x["hypotheses"]) <= 1:
        return x["hypotheses"]
    sorted_indices = np.argsort(x["predictions"])

    # First get the top 25 % indices
    top = ceil(p * len(x["predictions"]))
    top_indices = sorted_indices[-top:]

    # Get the ground truth score
    top_n = np.array(x["ground_truth"])[top_indices]

    # Get the best hypothesis out of it
    best = np.max(top_n)
    best_index = np.where(x["ground_truth"] == best)[0]
    best = x["hypotheses"][best_index].tolist()[0]

    return best

def get_top_p_predictions(val_df, p):
    val_df["ground_truth"] = val_df["utility"]
    top_p = val_df.apply(lambda x: map_to_top_p(x, p), axis=1).astype(str).to_list()
    return top_p

def map_to_highest_scoring(x):
    max_index = np.argmax(x["predictions"])
    best = x["hypotheses"][max_index]

    return best

def get_highest_scoring_predictions(predictions_df):
    return predictions_df.apply(map_to_highest_scoring, axis=1)






def get_kendall_taus(val_df):
    predictions_lists = val_df["predictions"].to_list()


    true_values_lists = val_df["utility"].to_list()


    one_hyp_count = 0
    kendall_taus = []
    lenghts = []
    for predictions, true_values in tqdm(zip(predictions_lists, true_values_lists), total=len(predictions_lists)):


        if len(predictions) == 1:
            one_hyp_count += 1
        else:

            kendall_tau = kendalltau(predictions, true_values).correlation
            if not math.isnan(kendall_tau):
                kendall_taus.append(kendall_tau)
                lenghts.append(len(predictions))
            else:
                # If nan then there is no correlation (predictions all have the same value) so set to 0
                kendall_taus.append(0)
                lenghts.append(len(predictions))
    return kendall_taus, lenghts, one_hyp_count


def plot_kendall_taus(val_df, save_location, pretty_name):

    kendall_taus, lengths, one_hyp_count = get_kendall_taus(val_df)

    fig = plt.figure()
    plt.title("Distribution of Kendall Taus for {}".format(pretty_name))
    plt.xlabel('Kendall Tau')
    sns.histplot(kendall_taus, )
    plt.axvline(x=np.mean(kendall_taus),
                color='red')
    plt.axvline(x=np.median(kendall_taus),
                color='red',
                ls='--'
                )
    save_loc = save_location + 'kendal_taus.png'
    plt.show()
    fig.savefig(save_loc, dpi=fig.dpi)


    fig = plt.figure()
    save_loc = save_location + 'kendal_taus_vs_length.png'
    plt.title("Kendall Tau vs Number of Hypotheses for {}".format(pretty_name))
    plt.xlabel("Number of hypotheses")
    plt.ylabel("Kendall Tau")
    plt.plot(lengths, kendall_taus, '.')

    plt.show()
    fig.savefig(save_loc, dpi=fig.dpi)
    return kendall_taus, lengths, one_hyp_count





def evaluate_predictions(val_df, model_name, pretty_name, utility="comet"):
    matplotlib.use('Agg')
    summary = {

    }


    result_ref = './results/{}/{}/'.format(utility, model_name)

    Path(result_ref).mkdir(parents=True, exist_ok=True)



    plot_rank_vs_predicted_util(val_df, save_location=result_ref, n_plots=5)

    kendall_taus, lengths, one_hyp_count = plot_kendall_taus(val_df, result_ref, pretty_name)


    summary["mean_kendall_taus"] = np.mean(kendall_taus)
    summary["median_kendall_taus"] = np.median(kendall_taus)
    summary["std_kendall_taus"] = np.std(kendall_taus)

    summary["MSE"] = get_mse(val_df)

    # Next we get the top 10 and highest scoring according to our model and use that as a reference.
    top_10_percent_prediction = get_top_p_predictions(val_df, 0.1)

    best_predictions = get_highest_scoring_predictions(val_df)

    # Evaluate the scores
    targets = val_df["target"].to_list()
    sources = val_df["source"].to_list()



    # First calculate the berstscores:




    bleurt_metric = BleurtUtility()

    bert_scores_best = bleurt_metric.evaluate(sources, best_predictions.to_list(), targets)

    summary["best_bleurt_mean"] = np.mean(bert_scores_best)
    summary["best_bleurt_median"] = np.median(bert_scores_best)
    summary["best_bleurt_std"] = np.std(bert_scores_best)

    top_10_bert_scores = bleurt_metric.evaluate(sources, top_10_percent_prediction, targets)

    summary["top_10_bleurt_mean"] = np.mean(top_10_bert_scores)
    summary["top_10_bleurt_median"] = np.median(top_10_bert_scores)
    summary["top_10_bleurt_std"] = np.std(top_10_bert_scores)



    if utility == 'comet':
        # Load the comet model


        # Get the comet score
        model_path = download_model("wmt20-comet-da")
        model = load_from_checkpoint(model_path)

        model.to("cuda")
        model.eval()
        wrapped_model = CometWrapper(model)

        top_10_comet_scores = wrapped_model.batch_predict(sources, top_10_percent_prediction, targets)

        # sns.histplot(top_10_comet_scores)

        summary["top_10_comet_mean"] = np.mean(top_10_comet_scores)
        summary["top_10_comet_median"] = np.median(top_10_comet_scores)
        summary["top_10_comet_std"] = np.std(top_10_comet_scores)

        best_predictions_comet_scores = wrapped_model.batch_predict(sources, best_predictions, targets)

        summary["best_comet_mean"] = np.mean(best_predictions_comet_scores)
        summary["best_comet_median"] = np.median(best_predictions_comet_scores)
        summary["best_comet_std"] = np.std(best_predictions_comet_scores)

    elif utility == 'unigram-f1':
        # Load the comet model


        # Get the comet score
        utility = NGramF(n=1)

        top_10_comet_scores = utility.evaluate(sources, top_10_percent_prediction, targets)

        # sns.histplot(top_10_comet_scores)

        summary["top_10_unigram_f1_mean"] = np.mean(top_10_comet_scores)
        summary["top_10_unigram_f1_median"] = np.median(top_10_comet_scores)
        summary["top_10_unigram_f1_std"] = np.std(top_10_comet_scores)

        best_predictions_comet_scores = utility.evaluate(sources, best_predictions, targets)

        summary["best_unigram_f1_mean"] = np.mean(best_predictions_comet_scores)
        summary["best_unigram_f1_median"] = np.median(best_predictions_comet_scores)
        summary["best_unigram_f1_std"] = np.std(best_predictions_comet_scores)
    elif utility == 'chrf':
        # Load the comet model


        # Get the comet score
        utility = ChrF()

        top_10_comet_scores = utility.evaluate(sources, top_10_percent_prediction, targets)

        # sns.histplot(top_10_comet_scores)

        summary["top_10_chrf_mean"] = np.mean(top_10_comet_scores)
        summary["top_10_chrf_median"] = np.median(top_10_comet_scores)
        summary["top_10_chrf_std"] = np.std(top_10_comet_scores)

        best_predictions_comet_scores = utility.evaluate(sources, best_predictions, targets)

        summary["best_chrf_mean"] = np.mean(best_predictions_comet_scores)
        summary["best_chrf_median"] = np.median(best_predictions_comet_scores)
        summary["best_chrf_std"] = np.std(best_predictions_comet_scores)


    else:
        raise 'utility not found: {}'.format(utility)


    # Save the summary
    summary_ref = result_ref + 'summary.json'

    with open(summary_ref, 'w') as fp:
        json.dump(summary, fp)