# -*- coding:utf-8 -*-
"""
作者：DELL
日期：2023年10月23日
"""
from sklearn import metrics
import torch
import torch.nn.functional as F


###################################################################
###################################################################
#                         分类
###################################################################
###################################################################


def acc(scores, labels):
    """
    Computes the accuracy of a model's predictions.

    Args:
        scores (torch.Tensor): A 2D tensor of predicted scores.
        labels (torch.Tensor): A 1D tensor of true labels.

    Returns:
        float: The accuracy of the model. The higher the better. Its range is [0, 1].
    """
    return (scores.argmax(-1) == labels).float().mean()


def precision(scores, labels, binary=False):
    """
    Calculates the precision score for a given set of predictions and labels.

    Args:
        scores: 2D torch.tensor containing the predicted scores for each class.
        labels: 1D tensor containing the true labels for each prediction.
        binary: bool indicating whether to calculate precision for binary classification or not.

    Returns:
        precision score as a float: The higher the better. Its range is [0, 1].
    """
    if binary:
        s = metrics.precision_score(labels, scores.argmax(-1), average='binary', zero_division=0.0)
    else:
        s = metrics.precision_score(labels, scores.argmax(-1), average='macro', zero_division=0.0)
    return s


def recall(scores, labels, binary=False):
    """
    Calculate recall score.

    Args:
        scores (torch.Tensor): A 2D tensor of predicted scores.
        labels (torch.Tensor): A 1D tensor of true labels.
        binary (bool, optional): If True, calculate binary recall. Default is False.

    Returns:
        float: The recall score: The higher the better. Its range is [0, 1].
    """
    if binary:
        s = metrics.recall_score(labels, scores.argmax(-1), average='binary', zero_division=0.0)
    else:
        s = metrics.recall_score(labels, scores.argmax(-1), average='macro', zero_division=0.0)
    return s


def f1(scores, labels, binary=False):
    """
    Calculate the F1 score.

    Args:
        scores (torch.Tensor): 2D tensor of predicted scores.
        labels (torch.Tensor): 1D tensor of true labels.
        binary (bool): Whether to calculate the F1 score for binary classification or not.

    Returns:
        float: The F1 score: The higher the better. Its range is [0, 1].
    """
    if binary:
        s = metrics.f1_score(labels, scores.argmax(-1), average='binary', zero_division=0.0)
    else:
        s = metrics.f1_score(labels, scores.argmax(-1), average='macro', zero_division=0.0)
    return s


def get_topk(topk, return_correct_num=False):
    """
    Returns a function that calculates the top-k accuracy metric for a given k.

    Args:
        topk (int): The value of k for which the top-k accuracy metric is to be calculated. Must be >= 1.
        return_correct_num (bool): Whether to return the number of correct predictions instead of the accuracy score.

    Returns:
        A function that calculates the top-k accuracy metric for a given set of scores and labels.
    """
    def topk_metric(scores, labels):
        """
        Computes the top-k accuracy score for a given set of predicted scores and true labels.

        Args:
            scores (torch.Tensor): A 2D tensor of predicted scores.
            labels (torch.Tensor): A 1D tensor of true labels.

        Returns:
            The top-k accuracy score.
        """
        labels = labels.detach().numpy()
        scores = scores.detach().numpy()
        if return_correct_num:
            num = metrics.top_k_accuracy_score(labels, scores, k=topk, normalize=False, labels=range(scores.shape[-1]))
            return num
        else:
            s = metrics.top_k_accuracy_score(labels, scores, k=topk, normalize=True, labels=range(scores.shape[-1]))
            return s

    setattr(topk_metric, '__qualname__', 'top' + str(topk))
    return topk_metric


def matthews_corrcoef(scores, labels):
    """
    Computes the Matthews correlation coefficient (MCC) for a given set of predicted scores and true labels.

    Args:
        scores (torch.Tensor): A 2D tensor of predicted scores.
        labels (torch.Tensor): A 1D tensor of true labels.

    Returns:
        The MCC score: The higher the better. Its range is [-1, 1].
    """
    scores = scores.argmax(-1)
    return metrics.matthews_corrcoef(labels, scores)


def hamming_loss(scores, labels):
    """
    Computes the hamming loss for a given set of predicted scores and true labels.

    Args:
        scores (torch.Tensor): A 2D tensor of predicted scores.
        labels (torch.Tensor): A 1D tensor of true labels.

    Returns:
        The hamming loss: The lower the better. Its range is [0, 1]
    """
    scores = scores.argmax(-1)
    return metrics.hamming_loss(labels, scores)


def zero_one_loss(scores, labels):
    """
    Computes the zero one loss for a given set of predicted scores and true labels.

    Args:
        scores (torch.Tensor): A 2D tensor of predicted scores.
        labels (torch.Tensor): A 1D tensor of true labels.

    Returns:
        The zero one loss: The lower the better. Its range is [0, 1]. Because normalize is True by default.
    """
    scores = scores.argmax(-1)
    return metrics.zero_one_loss(labels, scores)


def confusion_matrix(scores, labels):
    """
    Computes the confusion matrix for a given set of predicted scores and true labels.

    Args:
        scores (torch.Tensor): A 2D tensor of predicted scores.
        labels (torch.Tensor): A 1D tensor of true labels.

    Returns:
        The confusion matrix. The shape is (num_classes, num_classes).
    """
    scores = scores.argmax(-1)
    return metrics.confusion_matrix(labels, scores)