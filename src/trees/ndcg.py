import numpy as np
import collections

def validate(qids, targets, preds, k):
    """
    Predicts the scores for the test dataset and calculates the NDCG value.
    Parameters
    ----------
    data : Numpy array of documents
        Numpy array of documents with each document's format is [relevance score, query index, feature vector]
    k : int
        this is used to compute the NDCG@k

    Returns
    -------
    average_ndcg : float
        This is the average NDCG value of all the queries
    predicted_scores : Numpy array of scores
        This contains an array or the predicted scores for the documents.
    """
    query_groups = get_groups(qids)  # (qid,from,to),一个元组,表示这个qid的样本从哪到哪
    all_ndcg = []
    every_qid_ndcg = collections.OrderedDict()

    for qid, a, b in query_groups:
        predicted_sorted_indexes = np.argsort(preds[a:b])[::-1] # 从大到小的索引
        t_results = targets[a:b] # 目标数据的相关度
        t_results = t_results[predicted_sorted_indexes] #是predicted_sorted_indexes排好序的在test_data中的相关度

        dcg_val = dcg_k(t_results, k)
        idcg_val = ideal_dcg_k(t_results, k)
        ndcg_val = (dcg_val / idcg_val)
        all_ndcg.append(ndcg_val)
        every_qid_ndcg.setdefault(qid, ndcg_val)

    average_ndcg = np.nanmean(all_ndcg)
    return average_ndcg, every_qid_ndcg


    '''
    for query in query_indexes:
        results = np.zeros(len(query_indexes[query]))

        for tree in self.trees:
            results += self.learning_rate * tree.predict(data[query_indexes[query], 2:])
        predicted_sorted_indexes = np.argsort(results)[::-1]
        t_results = data[query_indexes[query], 0] # 第0列的相关度
        t_results = t_results[predicted_sorted_indexes]

        dcg_val = dcg_k(t_results, k)
        idcg_val = ideal_dcg_k(t_results, k)
        ndcg_val = (dcg_val / idcg_val)
        average_ndcg.append(ndcg_val)
    average_ndcg = np.nanmean(average_ndcg)
    return average_ndcg
'''

def get_groups(qids):
    """Makes an iterator of query groups on the provided list of query ids.

    Parameters
    ----------
    qids : array_like of shape = [n_samples]
        List of query ids.

    Yields
    ------
    row : (qid, int, int)
        Tuple of query id, from, to.
        ``[i for i, q in enumerate(qids) if q == qid] == range(from, to)``

    """
    prev_qid = None
    prev_limit = 0
    total = 0

    for i, qid in enumerate(qids):
        total += 1
        if qid != prev_qid:
            if i != prev_limit:
                yield (prev_qid, prev_limit, i)
            prev_qid = qid
            prev_limit = i

    if prev_limit != total:
        yield (prev_qid, prev_limit, total)

def group_queries(training_data, qid_index):
    """
        Returns a dictionary that groups the documents by their query ids.
        Parameters
        ----------
        training_data : Numpy array of lists
            Contains a list of document information. Each document's format is [relevance score, query index, feature vector]
        qid_index : int
            This is the index where the qid is located in the training data

        Returns
        -------
        query_indexes : dictionary
            The keys were the different query ids and teh values were the indexes in the training data that are associated of those keys.
    """
    query_indexes = {}  # 每个qid对应的样本索引范围,比如qid=1020,那么此qid在training data中的训练样本从0到100的范围, { key=str,value=[] }
    index = 0
    for record in training_data:
        query_indexes.setdefault(record[qid_index], [])
        query_indexes[record[qid_index]].append(index)
        index += 1
    return query_indexes


def dcg_k(scores, k):
    """
        Returns the DCG value of the list of scores and truncates to k values.
        Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order
        k : int
            In the amount of values you want to only look at for computing DCG

        Returns
        -------
        DCG_val: int
            This is the value of the DCG on the given scores
    """
    return np.sum([
                      (np.power(2, scores[i]) - 1) / np.log2(i + 2)
                      for i in range(len(scores[:k]))
                      ])


def ideal_dcg_k(scores, k):
    """
    前k个理想状态下的dcg
        Returns the Ideal DCG value of the list of scores and truncates to k values.
        Parameters
        ----------
        scores : list
            Contains labels in a certain ranked order
        k : int
            In the amount of values you want to only look at for computing DCG

        Returns
        -------
        Ideal_DCG_val: int
            This is the value of the Ideal DCG on the given scores
    """
    # 相关度降序排序
    scores = [score for score in sorted(scores)[::-1]]
    return dcg_k(scores, k)