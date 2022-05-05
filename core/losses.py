import torch


def sequence_loss(flow_preds, flow_gt, gamma=0.8, MAX_FLOW=400):
    """ Loss function defined over sequence of flow predictions """

    mag = torch.sum(flow_gt ** 2, dim=1).sqrt()
    Mask = torch.zeros([flow_gt.shape[0], flow_gt.shape[1], flow_gt.shape[2],
                        flow_gt.shape[3]]).cuda()
    mask = (flow_gt[:, 0, :, :] != 0) + (flow_gt[:, 1, :, :] != 0)
    valid = mask & (mag < MAX_FLOW)
    Mask[:, 0, :, :] = valid
    Mask[:, 1, :, :] = valid
    Mask = Mask != 0
    mask_sum = torch.sum(mask, dim=[1, 2])

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        Loss_reg = (flow_preds[i] - flow_gt) * Mask
        Loss_reg = torch.norm(Loss_reg, dim=1)
        Loss_reg = torch.sum(Loss_reg, dim=[1, 2])
        Loss_reg = Loss_reg / mask_sum
        flow_loss += i_weight * Loss_reg.mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics