import torch
import torch.distributed as distributed

from panoptic_bev.utils.sequence import pad_packed_images


def panoptic_stats(msk_gt, cat_gt, panoptic_pred, num_classes, _num_stuff):
    # Move gt to CPU
    msk_gt, cat_gt = msk_gt.cpu(), cat_gt.cpu()
    msk_pred, cat_pred, _, iscrowd_pred = panoptic_pred
    
    # Convert crowd predictions to void
    msk_remap = msk_pred.new_zeros(cat_pred.numel())
    msk_remap[~(iscrowd_pred > 0)] = torch.arange(0, (~(iscrowd_pred>0)).long().sum().item(), dtype=msk_remap.dtype,
                                                  device=msk_remap.device)

    msk_pred = msk_remap[msk_pred]
    cat_pred = cat_pred[~(iscrowd_pred>0)]

    iou = msk_pred.new_zeros(num_classes, dtype=torch.double)
    tp = msk_pred.new_zeros(num_classes, dtype=torch.double)
    fp = msk_pred.new_zeros(num_classes, dtype=torch.double)
    fn = msk_pred.new_zeros(num_classes, dtype=torch.double)

    if cat_gt.numel() > 1:
        msk_gt = msk_gt.view(-1)
        msk_pred = msk_pred.view(-1)

        # Compute confusion matrix
        confmat = msk_pred.new_zeros(cat_gt.numel(), cat_pred.numel(), dtype=torch.double)
        confmat.view(-1).index_add_(0, msk_gt * cat_pred.numel() + msk_pred,
                                    confmat.new_ones(msk_gt.numel()))

        # track potentially valid FP, i.e. those that overlap with void_gt <= 0.5
        num_pred_pixels = confmat.sum(0)
        valid_fp = (confmat[0] / num_pred_pixels) <= 0.5

        # compute IoU without counting void pixels (both in gt and pred)
        _iou = confmat / ((num_pred_pixels - confmat[0]).unsqueeze(0) + confmat.sum(1).unsqueeze(1) - confmat)

        # flag TP matches, i.e. same class and iou > 0.5
        matches = ((cat_gt.unsqueeze(1) == cat_pred.unsqueeze(0)) & (_iou > 0.5))

        # remove potential match of void_gt against void_pred
        matches[0, 0] = 0

        _iou = _iou[matches]
        tp_i, _ = matches.max(1)
        fn_i = ~tp_i
        fn_i[0] = 0  # remove potential fn match due to void against void
        fp_i = ~matches.max(0)[0] & valid_fp
        fp_i[0] = 0  # remove potential fp match due to void against void

        # Compute per instance classes for each tp, fp, fn
        tp_cat = cat_gt[tp_i]
        fn_cat = cat_gt[fn_i]
        fp_cat = cat_pred[fp_i]

        # Accumulate per class counts
        if tp_cat.numel() > 0:
            tp.index_add_(0, tp_cat, tp.new_ones(tp_cat.numel()))
        if fp_cat.numel() > 0:
            fp.index_add_(0, fp_cat, fp.new_ones(fp_cat.numel()))
        if fn_cat.numel() > 0:
            fn.index_add_(0, fn_cat, fn.new_ones(fn_cat.numel()))
        if tp_cat.numel() > 0:
            iou.index_add_(0, tp_cat, _iou)

    # note else branch is not needed because if cat_gt has only void we don't penalize predictions
    return iou, tp, fp, fn


def panoptic_post_processing(result, idx, msk, cat, iscrowd):
    panoptic_pred_list = []
    for i, (idy, po_pred, po_class, po_iscrowd, sem_pred, bbx_pred, cls_pred, obj_pred, msk_pred, sem_logits, msk_gt,
            cat_gt, iscrowd) in enumerate(zip(idx, result['po_pred'], result['po_class'], result['po_iscrowd'],
                                              result["sem_pred"], result["bbx_pred"], result["cls_pred"],
                                              result["obj_pred"], result["msk_pred"], result["sem_logits"], msk, cat,
                                              iscrowd)):
        msk_gt = msk_gt.squeeze(0)
        sem_gt = cat_gt[msk_gt]

        # Remove crowd from gt
        cmap = msk_gt.new_zeros(cat_gt.numel())
        cmap[~(iscrowd > 0)] = torch.arange(0, (~(iscrowd > 0)).long().sum().item(), dtype=cmap.dtype, device=cmap.device)
        msk_gt = cmap[msk_gt]
        cat_gt = cat_gt[~(iscrowd > 0)]

        # Compute panoptic output
        panoptic_pred_list.append({"po_pred": (po_pred, po_class, None, po_iscrowd),
                                   "msk_gt": msk_gt,
                                   "sem_gt": sem_gt,
                                   "cat_gt": cat_gt,
                                   "idx": idx[i]})

    return panoptic_pred_list


def confusion_matrix(gt, pred):
    conf_mat = gt.new_zeros(256 * 256, dtype=torch.float)
    conf_mat.index_add_(0, gt.view(-1) * 256 + pred.view(-1), conf_mat.new_ones(gt.numel()))
    return conf_mat.view(256, 256)


def compute_panoptic_test_metrics(panoptic_pred_list, panoptic_buffer, conf_mat, **varargs):

    for i, po_dict in enumerate(panoptic_pred_list):
        sem_gt = po_dict['sem_gt']
        msk_gt = po_dict['msk_gt']
        cat_gt = po_dict['cat_gt']
        idx = po_dict['idx']
        panoptic_pred = po_dict['po_pred']

        panoptic_buffer += torch.stack(panoptic_stats(msk_gt, cat_gt, panoptic_pred, varargs['num_classes'],
                                                      varargs['num_stuff']), dim=0)

        # Calculate confusion matrix on panoptic output
        sem_pred = panoptic_pred[1][panoptic_pred[0]]

        conf_mat_i = confusion_matrix(sem_gt.cpu(), sem_pred)
        conf_mat += conf_mat_i.to(conf_mat)

    return panoptic_buffer, conf_mat


def get_panoptic_scores(panoptic_buffer, scores_out, device, num_stuff, debug):
    # Gather from all workers
    panoptic_buffer = panoptic_buffer.to(device)
    if not debug:
        distributed.all_reduce(panoptic_buffer, distributed.ReduceOp.SUM)

    # From buffers to scores
    denom = panoptic_buffer[1] + 0.5 * (panoptic_buffer[2] + panoptic_buffer[3])
    denom[denom == 0] = 1.
    scores = panoptic_buffer[0] / denom
    RQ = panoptic_buffer[1] / denom
    panoptic_buffer[1][panoptic_buffer[1] == 0] = 1.
    SQ = panoptic_buffer[0] / panoptic_buffer[1]

    scores_out["pq"] = scores.mean()
    scores_out["pq_stuff"] = scores[:num_stuff].mean()
    scores_out["pq_thing"] = scores[num_stuff:].mean()
    scores_out["sq"] = SQ.mean()
    scores_out["sq_stuff"] = (SQ[:num_stuff]).mean()
    scores_out["sq_thing"] = (SQ[num_stuff:]).mean()
    scores_out["rq"] = RQ.mean()
    scores_out["rq_stuff"] = (RQ[:num_stuff]).mean()
    scores_out["rq_thing"] = (RQ[num_stuff:]).mean()

    return scores_out


def make_panoptic_gt_list(msk, cat, iscrowd):
    msk_up, _ = pad_packed_images(msk)

    panoptic_gt_list = []
    for b in range(msk_up.shape[0]):
        panoptic_gt_list.append((msk_up[b, 0, :, :], cat[b], [], iscrowd[b]))

    return panoptic_gt_list


def make_semantic_gt_list(msk, cat):
    sem_out = []
    for msk_i, cat_i in zip(msk, cat):
        msk_i = msk_i.squeeze(0)
        sem_out.append(cat_i[msk_i])
    return sem_out
