import torch
import torch.nn as nn
import numpy as np

def get_batch_jacobian(net, x):
    """
    Computes the Jacobian of network outputs with respect to the input.
    """
    net.zero_grad()
    x.requires_grad_(True)

    _, y = net(x)
    y.backward(torch.ones_like(y))

    jacob = x.grad.detach()
    return jacob

def eval_score_perclass(jacob, labels=None, n_classes=10):
    """
    Computes Jacobian Covariance Score per class and aggregates.
    """
    k = 1e-5
    per_class = {}
    for i, label in enumerate(labels):
        if label in per_class:
            per_class[label] = np.vstack((per_class[label], jacob[i]))
        else:
            per_class[label] = jacob[i]

    ind_corr_matrix_score = {}
    for c in per_class.keys():
        try:
            corrs = np.corrcoef(per_class[c])
            s = np.sum(np.log(np.abs(corrs) + k))
            if n_classes > 100:
                s /= len(corrs)
            ind_corr_matrix_score[c] = s
        except Exception:
            continue

    score = 0
    keys = ind_corr_matrix_score.keys()
    if n_classes <= 100:
        for c in keys:
            score += np.abs(ind_corr_matrix_score[c])
    else:
        for c in keys:
            for cj in keys:
                score += np.abs(ind_corr_matrix_score[c] - ind_corr_matrix_score[cj])
        score /= len(keys)

    return score

def compute_jacov_score(network, gpu, trainloader, resolution, batch_size):
    """
    Computes Jacobian Covariance zero-cost proxy score.
    """
    if gpu is not None:
        device = torch.device(f"cuda:{gpu}")
    else:
        device = torch.device("cpu")

    network.to(device)
    network.eval()

    data_iter = iter(trainloader)
    x, target = next(data_iter)
    x, target = x.to(device), target.cpu().tolist()

    jacobs_batch = get_batch_jacobian(network, x)
    jacobs_batch = jacobs_batch.reshape(jacobs_batch.size(0), -1).cpu().numpy()

    try:
        score = eval_score_perclass(jacobs_batch, target)
    except Exception as e:
        print("Jacobian score error:", e)
        score = np.nan

    return {"jacov": score}
