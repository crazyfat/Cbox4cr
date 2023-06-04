import torch.nn.functional as F


def bpr_loss(positive_logit, negative_logit, theta):
    negative_score = F.logsigmoid(-negative_logit).mean(dim=-1)
    positive_score = F.logsigmoid(positive_logit)
    positive_sample_loss = - positive_score.sum()
    negative_sample_loss = - negative_score.sum()
    return (positive_sample_loss + negative_sample_loss) * theta
