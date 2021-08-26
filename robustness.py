"""
Perform Robustness Evaluation based on the work "Robustness Gym"
"""
import numpy as np
import pandas as pd
from nltk import tokenize
import argparse
from rouge_score import rouge_scorer, scoring
from scipy.stats import ttest_ind

from sentence_splitter import add_newline_to_end_of_each_sentence

parser = argparse.ArgumentParser()
parser.add_argument("--bart_file", type=str, default='./output/xsum-original/test_generations.txt')
parser.add_argument("--pred_file", type=str, default='./output/xsum/test_generations.txt')
parser.add_argument("--src_file", type=str, default='./data/xsum/test.source')
parser.add_argument("--tgt_file", type=str, default='./data/xsum/test.target')

args = parser.parse_args()

# baseline summary - BART
bart = [x.rstrip() for x in open(args.bart_file, encoding='utf8').readlines()]

# predict summary
preds = [x.rstrip() for x in open(args.pred_file, encoding='utf8').readlines()]

# target article
articles = [x.rstrip() for x in open(args.src_file, encoding='utf8').readlines()]

# target summary
targets = [x.rstrip() for x in open(args.tgt_file, encoding='utf8').readlines()]

assert len(preds) == len(articles)


def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {stat: round(getattr(mid, stat), 2) for stat in ["precision", "recall", "fmeasure"]}
    return new_dict


def calculate_rouge(
        pred_lns,
        tgt_lns,
        use_stemmer=True,
        rouge_keys=["rouge1", "rouge2", "rougeL"],
        return_precision_and_recall=False,
        bootstrap_aggregation=True,
        newline_sep=True,
):
    scorer = rouge_scorer.RougeScorer(rouge_keys, use_stemmer=use_stemmer)
    aggregator = scoring.BootstrapAggregator()
    for pred, tgt in zip(tgt_lns, pred_lns):
        # rougeLsum expects "\n" separated sentences within a summary
        if newline_sep:
            pred = add_newline_to_end_of_each_sentence(pred)
            tgt = add_newline_to_end_of_each_sentence(tgt)
        scores = scorer.score(pred, tgt)
        aggregator.add_scores(scores)
        # print(f"aggregator._scores: {aggregator._scores}")

    if bootstrap_aggregation:
        result = aggregator.aggregate()
        if return_precision_and_recall:
            return extract_rouge_mid_statistics(result)  # here we return dict
        else:
            return {k: round(v.mid.fmeasure * 100, 2) for k, v in result.items()}

    else:
        return aggregator._scores


def length(preds, articles, targets, percentile=0.1):
    """
    Create slices based on the length of the source document
    :return:
    """
    result = []
    for i in range(len(preds)):
        pred = preds[i]
        article = articles[i]
        target = targets[i]
        result.append({'article': article, 'pred': pred, 'target': target, 'length': len(article)})
    # sort the dataset based on length - ascending
    df = pd.DataFrame(result)
    df.sort_values(by=['length'], ascending=True, ignore_index=True, inplace=True)
    # create the subpopulations based on percentile
    shortest_article, longest_article, shortest_summary, longest_summary, shortest_target, longest_target = [], [], [], [], [], []

    for i in range(int(len(df) * percentile)):
        shortest_article.append(df['article'][i])
        shortest_summary.append(df['pred'][i])
        shortest_target.append(df['target'][i])
    for i in range(int(len(df) * (1 - percentile)) + 1, len(df)):
        longest_article.append(df['article'][i])
        longest_summary.append(df['pred'][i])
        longest_target.append(df['target'][i])

    # calculate the performance - ROUGE score
    metrics = calculate_rouge(pred_lns=shortest_summary, tgt_lns=shortest_target)
    print(f"Result for Shortest: {metrics}")
    metrics = calculate_rouge(pred_lns=longest_summary, tgt_lns=longest_target)
    print(f"Result for Longest: {metrics}")
    # calculate the total number of observations in two slices
    print(f"There are {len(shortest_article)} observations in Shortest Articles")
    print(f"There are {len(longest_article)} observations in Longest Articles")


def abstractiveness(preds, articles, targets, rouge_key='rouge1', percentile=0.1):
    """
    The degree to which the reference summary is abstractive versus extractive,
    based on the proportion of n-grams in the reference summary that are not in the article.

    abstractiveess(A,S) = 1-rouge_precision(A,S)

    rouge_key: can be the value of ["rouge1", "rouge2", "rougeL"], default value is 'rouge1'
    :return:
    """
    print(f"Evaluate the robustness through abstractiveness with {rouge_key}")
    result = []
    for i in range(len(articles)):
        article = articles[i]
        pred = preds[i]
        target = targets[i]
        # rouge_precision equals the proportion of n-grams in the reference summary that are also in the article
        metrics = calculate_rouge(pred_lns=[target], tgt_lns=[article], rouge_keys=[rouge_key],
                                  return_precision_and_recall=True)
        result.append({'article': article, 'pred': pred, 'target': target,
                       'abstractiveness': 1 - metrics[rouge_key]['precision']})

    df = pd.DataFrame(result)
    df.sort_values(by=['abstractiveness'], ascending=True, inplace=True, ignore_index=True)

    shortest_article, longest_article, shortest_summary, longest_summary, shortest_target, longest_target = [], [], [], [], [], []

    for i in range(int(len(df) * percentile)):
        shortest_article.append(df['article'][i])
        shortest_summary.append(df['pred'][i])
        shortest_target.append(df['target'][i])
    for i in range(int(len(df) * (1 - percentile)), len(df)):
        longest_article.append(df['article'][i])
        longest_summary.append(df['pred'][i])
        longest_target.append(df['target'][i])

    # for BART

    metrics = calculate_rouge(pred_lns=shortest_summary, tgt_lns=shortest_target,
                              rouge_keys=["rouge1", "rouge2", "rougeL"])
    print(f"Result for Least Abstractive: {metrics}")

    metrics = calculate_rouge(pred_lns=longest_summary, tgt_lns=longest_target,
                              rouge_keys=["rouge1", "rouge2", "rougeL"])
    print(f"Result for Most Abstractive: {metrics}")

    # calculate the total number of observations in two slices
    print(f"There are {len(shortest_article)} observations in Least Abstractive")
    print(f"There are {len(longest_article)} observations in Most Abstractive")


def distillation(preds, articles, targets, rouge_key='rouge1', percentile=0.1):
    """
    The degree to which the reference summary is distilled from a larger quantity of content,
    based on the proportion of n-grams in the article that do not appear in the reference summary.

    distillation(A,S) = 1-rouge_recall(A,S)
    :return:
    """
    print(f"Evaluate the robustness through distillation with {rouge_key}")
    result = []
    for i in range(len(articles)):
        article = articles[i]
        pred = preds[i]
        target = targets[i]
        # rouge_precision equals the proportion of n-grams in the reference summary that are also in the article
        metrics = calculate_rouge(pred_lns=[target], tgt_lns=[article], rouge_keys=[rouge_key],
                                  return_precision_and_recall=True)
        result.append(
            {'article': article, 'pred': pred, 'target': target,
             'distill': 1 - metrics[rouge_key]['recall']})

    df = pd.DataFrame(result)
    df.sort_values(by=['distill'], ascending=True, inplace=True, ignore_index=True)

    # create the subpopulations based on percentile
    shortest_article, longest_article, shortest_summary, longest_summary, shortest_target, longest_target = [], [], [], [], [], []

    for i in range(int(len(df) * percentile)):
        shortest_article.append(df['article'][i])
        shortest_summary.append(df['pred'][i])
        shortest_target.append(df['target'][i])
    for i in range(int(len(df) * (1 - percentile)), len(df)):
        longest_article.append(df['article'][i])
        longest_summary.append(df['pred'][i])
        longest_target.append(df['target'][i])

    # for BART

    metrics = calculate_rouge(pred_lns=shortest_summary, tgt_lns=shortest_target,
                              rouge_keys=["rouge1", "rouge2", "rougeL"])
    print(f"Result for Least Distilled: {metrics}")

    metrics = calculate_rouge(pred_lns=longest_summary, tgt_lns=longest_target,
                              rouge_keys=["rouge1", "rouge2", "rougeL"])
    print(f"Result for Most Distilled: {metrics}")

    # calculate the total number of observations in two slices
    print(f"There are {len(shortest_article)} observations in Least Abstractive")
    print(f"There are {len(longest_article)} observations in Most Abstractive")


def position(preds, articles, targets, rouge_key='rouge1', percentile=0.1):
    print(f"Evaluate the robustness through position with {rouge_key}")
    result = []
    for i in range(len(articles)):
        # input document
        article = articles[i]
        # prediction; generated summary
        pred = preds[i]
        # target summary
        target = targets[i]
        # find the position of the best matched summary in the document
        sentences = tokenize.sent_tokenize(article)
        sent_sim = []
        for j in range(len(sentences)):
            metrics = calculate_rouge(pred_lns=[sentences[j]], tgt_lns=[target], rouge_keys=[rouge_key],
                                      return_precision_and_recall=True)
            sent_sim.append(metrics[rouge_key]['fmeasure'])
        # record the best-matched sentence (index is from 0)
        result.append(
            {'article': article, 'pred': pred, 'target': target, 'best-matched': sent_sim.index(max(sent_sim))})
    # average
    df = pd.DataFrame(result)
    df.sort_values(by=['best-matched'], ascending=True, inplace=True, ignore_index=True)

    shortest_article, longest_article, shortest_summary, longest_summary, shortest_target, longest_target = [], [], [], [], [], []

    for i in range(int(len(df) * percentile)):
        shortest_article.append(df['article'][i])
        shortest_summary.append(df['pred'][i])
        shortest_target.append(df['target'][i])
    for i in range(int(len(df) * (1 - percentile)), len(df)):
        longest_article.append(df['article'][i])
        longest_summary.append(df['pred'][i])
        longest_target.append(df['target'][i])
    # evaluate the performance - ROUGE
    metrics = calculate_rouge(pred_lns=shortest_summary, tgt_lns=shortest_target,
                              rouge_keys=["rouge1", "rouge2", "rougeL"])
    print(f"Result for Earliest Position: {metrics}")

    metrics = calculate_rouge(pred_lns=longest_summary, tgt_lns=longest_target,
                              rouge_keys=["rouge1", "rouge2", "rougeL"])
    print(f"Result for Latest Position: {metrics}")

    # calculate the total number of observations in two slices
    print(f"There are {len(shortest_article)} observations in Earliest Position")
    print(f"There are {len(longest_article)} observations in Latest Position")


# file: list, the first element is from bart, and the second is from elsa
file = [bart, preds]

for i in range(2):
    if i < 1:
        print(f"Robustness Result for BART")
    else:
        print(f"Robustness Result for TAAS")
    length(file[i], articles, targets, percentile=0.1)
    abstractiveness(file[i], articles, targets, percentile=0.1)
    distillation(file[i], articles, targets, percentile=0.1)
    position(file[i], articles, targets, percentile=0.1)
