import glob
import os
import xml.etree.ElementTree as et
import operator
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.summarization.summarizer import summarize
from gensim.summarization import keywords as gensim_keywords
from improved_summarizer import set_embedding_model as s_set_embedding_model, \
    summarize as improved_summarize, query_based_summarization_baseline
from improved_keywords import set_embedding_model as k_set_embedding_model, \
    keywords as improved_keywords
from rouge_score import rouge_scorer
from w2v import WordEmbeddingModel
from tabulate import tabulate

dir_path = os.path.dirname(os.path.realpath(__file__))
wp = WordEmbeddingModel()
s_set_embedding_model(wp.model)
k_set_embedding_model(wp.model)


def update_score(oldScore, newScore):
    if oldScore is None:
        oldScore = newScore
    else:
        for metric in newScore.keys():
            oldScore[metric] = tuple(map(operator.add, oldScore[metric], newScore[metric]))
    return oldScore


def get_score_avg(score, n_of_scores):
    for metric in score.keys():
        score[metric] = tuple(x / n_of_scores for x in score[metric])
    return score


def get_text_from_file(file_path):
    text = None
    with open(file_path, encoding="utf8", errors='ignore') as f:
        text = f.read()
    f.close()
    return text


def get_bbc_dataset_files():
    categories = ['business', 'entertainment', 'politics', 'tech', 'sport']
    data = []
    for c in categories:
        originals = glob.glob(os.path.join(dir_path,
                                           'nlp_data/evaluation_data/BBC News Summary/News Articles/' + c + '/*.txt'))
        summaries = glob.glob(os.path.join(dir_path,
                                           'nlp_data/evaluation_data/BBC News Summary/Summaries/' + c + '/*.txt'))
        originals.sort()
        summaries.sort()
        for i in range(len(originals)):
            original = get_text_from_file(originals[i])
            summary = get_text_from_file(summaries[i])
            data.append({'full': original, 'summary': summary})
    return data


def bbc_dataset_statistics():
    t_ratio = 0
    t_text = 0
    news = get_bbc_dataset_files()
    for n in news:
        text_size = len(word_tokenize(n['full']))
        summary_size = len(word_tokenize(n['summary']))
        t_ratio += (summary_size / text_size)
        t_text += text_size
    print('Average news size: ' + str(t_text / len(news)) + ' words')
    print('Average summary to text ratio: ' + str(t_ratio / len(news)))


def bbc_dataset_rouge():
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    test_data = get_bbc_dataset_files()
    generic_baseline, generic_improved, generic_improved_redundancy_removal = None, None, None
    for v in test_data:
        text = v['full']
        summary = v['summary']
        scores = scorer.score(summarize(text, ratio=0.4), summary)
        generic_baseline = update_score(generic_baseline, scores)
        scores = scorer.score(improved_summarize(text, ratio=0.4, redundancy_removal=False), summary)
        generic_improved = update_score(generic_improved, scores)
        scores = scorer.score(improved_summarize(text, ratio=0.4, redundancy_removal=True), summary)
        generic_improved_redundancy_removal = update_score(generic_improved_redundancy_removal, scores)
    total_news = len(test_data)
    return {'generic_baseline': get_score_avg(generic_baseline, total_news),
            'generic_improved_redundancy_removal': get_score_avg(generic_improved_redundancy_removal, total_news),
            'generic_improved': get_score_avg(generic_improved, total_news)}


def get_email_dataset_files():
    stop_words = set(stopwords.words('english'))
    file_paths = [('CorporateSingleXML', 'CorporateSingle.xml'),
                  ('CorporateThreadXML', 'CorporateThread.xml'),
                  ('PrivateSingleXML', 'newPrivateSingle.xml'),
                  ('PrivateThreadXML', 'newPrivateThread.xml'),
                  ('PersonalSingleXML', 'PrivateSingle.xml'),
                  ('PersonalThreadXML', 'PrivateThread.xml')]
    data = {}
    for pair in file_paths:
        originals = glob.glob(os.path.join(dir_path,
                                           'nlp_data/evaluation_data/EmailSummarizationKeywordExtraction/' + pair[
                                               0] + '/*.xml'))
        annotations = os.path.join(dir_path,
                                   'nlp_data/evaluation_data/EmailSummarizationKeywordExtraction/Annotations/' + pair[
                                       1])
        for o in originals:
            xTree = et.parse(o)
            xRoot = xTree.getroot()
            id = xRoot.find(".//id").text
            text = {}
            for n in xRoot.findall(".//sentence"):
                if n.text is not None:
                    text[n.attrib.get("id")] = n.text
            data[id] = {'full': text}

        xTree = et.parse(annotations)
        xRoot = xTree.getroot()
        annotations = xRoot.findall('.//annotation')
        for a in annotations:
            id = a.attrib.get("email")
            summary = []
            keywords = []
            for s in a.findall('.//extractive_sentences/sentence'):
                summary.append(s.text)
            for k in a.findall('.//keyword_keyphrase/keyword'):
                if k.text is not None:
                    tokens = word_tokenize(k.text)
                    keywords.extend([w for w in tokens if not w in stop_words])
            if 'annotations' in data[id]:
                l = data[id]['annotations']
                l.append({'summary': summary, 'keywords': keywords})
            else:
                data[id]['annotations'] = [{'summary': summary, 'keywords': keywords}]
    return data


def email_dataset_statistics():
    # ratio of summaries to original text avg no. of keywords
    t_text = 0
    t_ratio = 0
    n_keywords = 0
    n_annotations = 0
    emails = get_email_dataset_files()
    for email in emails.values():
        text_size = 0
        for s in email['full'].values():
            text_size += len(word_tokenize(s))
        t_text += text_size
        for annotation in email['annotations']:
            summary_size = 0
            n_annotations += 1
            for s_id in annotation['summary']:
                if s_id in email['full']:
                    summary_size += len(word_tokenize(email['full'][s_id]))
            t_ratio += (summary_size / text_size)
            n_keywords += len(annotation['keywords'])
    print('Average email size: ' + str(t_text / len(emails.keys())) + ' words')
    print('Average summary to text length ratio: ' + str(t_ratio / n_annotations))
    print('Average no. of keywords: ' + str(n_keywords / n_annotations))


def email_dataset_rouge():
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    test_data = get_email_dataset_files()
    generic_baseline, generic_improved, generic_improved_redundancy_removal, query_based_baseline, query_based_improved, \
    query_based_improved_redundancy_removal, keyword_baseline, keyword_semantic = \
        None, None, None, None, None, None, None, None
    n_annotation = 0
    for v in test_data.values():
        text = ''
        for s in v['full'].values():
            text += s
        for annotation in v['annotations']:
            keywords = annotation['keywords']
            summary = ''
            for s_id in annotation['summary']:
                if s_id in v['full']:
                    summary += v['full'][s_id]
            scores = scorer.score(summarize(text, ratio=0.5), summary)
            generic_baseline = update_score(generic_baseline, scores)
            scores = scorer.score(improved_summarize(text, ratio=0.5, redundancy_removal=False), summary)
            generic_improved = update_score(generic_improved, scores)
            scores = scorer.score(improved_summarize(text, ratio=0.5, redundancy_removal=True), summary)
            generic_improved_redundancy_removal = update_score(generic_improved_redundancy_removal, scores)
            scores = scorer.score(query_based_summarization_baseline(text, ratio=0.5, query=keywords), summary)
            query_based_baseline = update_score(query_based_baseline, scores)
            scores = scorer.score(improved_summarize(text, ratio=0.5, query=keywords, redundancy_removal=False),
                                  summary)
            query_based_improved = update_score(query_based_improved, scores)
            scores = scorer.score(improved_summarize(text, ratio=0.5, query=keywords, redundancy_removal=True), summary)
            query_based_improved_redundancy_removal = update_score(query_based_improved_redundancy_removal, scores)
            try:
                scores = scorer.score(gensim_keywords(text, words=len(keywords)), ' '.join(keywords))
            except IndexError:  # Gensim known issue https://github.com/RaRe-Technologies/gensim/issues/2598
                scores = scorer.score(gensim_keywords(text, ratio=(len(keywords) / len(word_tokenize(text)))),
                                      ' '.join(keywords))
            keyword_baseline = update_score(keyword_baseline, scores)
            try:
                scores = scorer.score(improved_keywords(text, words=len(keywords)), ' '.join(keywords))
            except IndexError:
                scores = scorer.score(improved_keywords(text, ratio=(len(keywords) / len(word_tokenize(text)))),
                                      ' '.join(keywords))
            keyword_semantic = update_score(keyword_semantic, scores)
            n_annotation += 1
    return {'generic_baseline': get_score_avg(generic_baseline, n_annotation),
            'generic_improved_redundancy_removal': get_score_avg(generic_improved_redundancy_removal, n_annotation),
            'generic_improved': get_score_avg(generic_improved, n_annotation),
            'query_based_baseline': get_score_avg(query_based_baseline, n_annotation),
            'query_based_improved_redundancy_removal': get_score_avg(query_based_improved_redundancy_removal,
                                                                     n_annotation),
            'query_based_improved': get_score_avg(query_based_improved, n_annotation),
            'keyword_baseline': get_score_avg(keyword_baseline, n_annotation),
            'keyword_semantic': get_score_avg(keyword_semantic, n_annotation)}


def pretty_print_resutls(scores):
    headers = ['*', 'ROUGE1-PR', 'ROUGE1-RC', 'ROUGE1-F1', 'ROUGE2-PR', 'ROUGE2-RC', 'ROUGE2-F1',
               'ROUGEL-PR', 'ROUGEL-RC', 'ROUGEL-F1']
    table = []
    for key, val in scores.items():
        row = [key]
        for metric in val.keys():
            row.extend([str(x) for x in val[metric]])
        table.append(row)

    print(tabulate(table, headers, tablefmt="grid"))


print('BBC News Dataset\n')
bbc_dataset_statistics()
pretty_print_resutls(bbc_dataset_rouge())
print('---------------------------------------------------------------------------------------\n')
print('Enron Email Dataset\n')
email_dataset_statistics()
pretty_print_resutls(email_dataset_rouge())
