import argparse
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

try:
    from tensorboardX import SummaryWriter
except ModuleNotFoundError:
    from torch.utils.tensorboard import SummaryWriter

from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from networks.net_factory import net_factory
from utils import losses, ramps
from val_2D import test_single_volume


parser = argparse.ArgumentParser(
    description='Train URRL for semi-supervised ultrasound segmentation.'
)
parser.add_argument('--root_path', type=str, required=True, help='dataset root')
parser.add_argument('--snapshot_path', type=str,
                    default='./runs/urrl',
                    help='directory for checkpoints and logs')
parser.add_argument('--exp', type=str, default='tn3k/URCCompact_URE', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='unet')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.02, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=int, nargs=2, default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--num_classes', type=int, default=2, help='output channel of network')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4, help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=float, default=0.1, help='labeled data ratio')

# cps costs
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')

# uncertainty-ranked contrastive loss
parser.add_argument('--urc_weight', type=float, default=0.4, help='weight of uncertainty-ranked contrastive loss')
parser.add_argument('--urc_start_iter', type=int, default=1000, help='start iteration of URC loss')
parser.add_argument('--urc_patch_size', type=int, default=16, help='patch pooling size for URC')
parser.add_argument('--urc_conf_thresh', type=float, default=0.75, help='confidence threshold for reliable patches')
parser.add_argument('--urc_entropy_thresh', type=float, default=0.25, help='entropy threshold for reliable patches')
parser.add_argument('--urc_disagree_thresh', type=float, default=0.10, help='disagreement threshold for reliable patches')
parser.add_argument('--urc_amb_entropy_thresh', type=float, default=0.55, help='entropy threshold for ambiguous patches')
parser.add_argument('--urc_amb_disagree_thresh', type=float, default=0.25, help='disagreement threshold for ambiguous patches')
parser.add_argument('--urc_margin_pos_mid', type=float, default=0.15, help='ranking margin between positive and ambiguous')
parser.add_argument('--urc_margin_mid_neg', type=float, default=0.10, help='ranking margin between ambiguous and negative')
parser.add_argument('--urc_max_samples_per_class', type=int, default=128, help='maximum reliable anchors per class in each batch')

# reliable class-center compactness
parser.add_argument('--urc_compact_weight', type=float, default=0.08, help='weight of reliable class-center compactness')
parser.add_argument('--urc_compact_start_iter', type=int, default=2000, help='start iteration of class-center compactness')

# uncertainty-guided region exchange
parser.add_argument('--ure_weight', type=float, default=0.2, help='weight of uncertainty-guided region exchange loss')
parser.add_argument('--ure_start_iter', type=int, default=3000, help='start iteration of URE loss')
parser.add_argument('--ure_conf_thresh', type=float, default=0.75, help='confidence threshold for reliable source region')
parser.add_argument('--ure_entropy_thresh', type=float, default=0.30, help='entropy threshold for reliable source region')
parser.add_argument('--ure_disagree_thresh', type=float, default=0.08, help='disagreement threshold for reliable source region')
parser.add_argument('--ure_amb_entropy_thresh', type=float, default=0.55, help='entropy threshold for ambiguous target region')
parser.add_argument('--ure_amb_disagree_thresh', type=float, default=0.20, help='disagreement threshold for ambiguous target region')
parser.add_argument('--ure_low_conf_thresh', type=float, default=0.55, help='low confidence threshold for unstable target region')
parser.add_argument('--ure_unstable_entropy_thresh', type=float, default=0.75, help='entropy threshold for unstable target region')
parser.add_argument('--ure_unstable_disagree_thresh', type=float, default=0.30, help='disagreement threshold for unstable target region')
parser.add_argument('--ure_amb_replace_ratio', type=float, default=0.5, help='replacement ratio for ambiguous regions')
parser.add_argument('--ure_block_size', type=int, default=16, help='block size for region-level exchange')
parser.add_argument('--ure_reliable_ratio', type=float, default=0.60, help='minimum reliable ratio inside a source block')
parser.add_argument('--ure_target_ratio', type=float, default=0.20, help='minimum ambiguous/unstable ratio inside a target block')
parser.add_argument('--ure_min_pixels', type=float, default=32.0, help='minimum exchanged pixels to enable URE')
parser.add_argument('--ure_use_dice', type=int, default=1, help='use dice loss for mixed pseudo label')
parser.add_argument('--ure_dice_weight', type=float, default=0.5, help='dice weight in URE mixed loss')

args = parser.parse_args()


def get_current_consistency_weight(epoch):
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def unwrap_output(out):
    if isinstance(out, dict):
        if 'out' in out:
            return out['out']
        if 'logits' in out:
            return out['logits']
        return next(iter(out.values()))
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


class PenultimateFeatureHook(object):
    def __init__(self, model, num_classes):
        self.model = model
        self.num_classes = num_classes
        self.feature = None
        self.handle = None
        self.target_module = self._find_target_module()
        if self.target_module is not None:
            self.handle = self.target_module.register_forward_pre_hook(self._hook_fn)

    def _find_target_module(self):
        candidate = None
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and getattr(module, 'out_channels', None) == self.num_classes:
                candidate = module
        return candidate

    def _hook_fn(self, module, inputs):
        if not inputs:
            self.feature = None
            return
        x = inputs[0]
        if torch.is_tensor(x) and x.dim() == 4:
            self.feature = x
        else:
            self.feature = None

    def forward(self, x):
        self.feature = None
        out = unwrap_output(self.model(x))
        feat = self.feature if self.feature is not None else out
        return out, feat

    def close(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def normalized_entropy(prob, eps=1e-8):
    num_classes = prob.shape[1]
    ent = -(prob * (prob + eps).log()).sum(dim=1)
    if num_classes > 1:
        ent = ent / np.log(num_classes)
    return ent


def avg_pool_patch(x, patch_size):
    if patch_size <= 1:
        return x
    h, w = x.shape[-2], x.shape[-1]
    k = min(patch_size, h, w)
    if k <= 1:
        return x
    return F.avg_pool2d(x, kernel_size=k, stride=k)


def flatten_patch_map(x):
    x = x.permute(0, 2, 3, 1).contiguous()
    return x.view(-1, x.shape[-1])


def flatten_scalar_map(x):
    return x.contiguous().view(-1)


def symmetric_kl(p, q, eps=1e-8):
    kl_pq = (p * ((p + eps).log() - (q + eps).log())).sum(dim=1)
    kl_qp = (q * ((q + eps).log() - (p + eps).log())).sum(dim=1)
    return 0.5 * (kl_pq + kl_qp)


def pixel_symmetric_kl(prob1, prob2, eps=1e-8):
    kl_12 = (prob1 * ((prob1 + eps).log() - (prob2 + eps).log())).sum(dim=1)
    kl_21 = (prob2 * ((prob2 + eps).log() - (prob1 + eps).log())).sum(dim=1)
    return 0.5 * (kl_12 + kl_21)


def sample_balanced_indices(mask, labels, max_samples_per_class, num_classes):
    device = labels.device
    selected = []
    for c in range(num_classes):
        idx = torch.where(mask & (labels == c))[0]
        if idx.numel() == 0:
            continue
        if idx.numel() > max_samples_per_class:
            perm = torch.randperm(idx.numel(), device=device)[:max_samples_per_class]
            idx = idx[perm]
        selected.append(idx)
    if not selected:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(selected, dim=0)


def build_patch_group(feat1, feat2, prob1, prob2, patch_size,
                      conf_thresh, ent_thresh, disagree_thresh,
                      amb_ent_thresh, amb_disagree_thresh):
    feat1_p = avg_pool_patch(feat1, patch_size)
    feat2_p = avg_pool_patch(feat2, patch_size)
    prob1_p = avg_pool_patch(prob1, patch_size)
    prob2_p = avg_pool_patch(prob2, patch_size)

    prob1_p = prob1_p / (prob1_p.sum(dim=1, keepdim=True) + 1e-6)
    prob2_p = prob2_p / (prob2_p.sum(dim=1, keepdim=True) + 1e-6)

    ent1 = normalized_entropy(prob1_p)
    ent2 = normalized_entropy(prob2_p)
    conf1, label1 = prob1_p.max(dim=1)
    conf2, label2 = prob2_p.max(dim=1)
    disagree = symmetric_kl(flatten_patch_map(prob1_p), flatten_patch_map(prob2_p)).view_as(ent1)

    same_label = label1 == label2
    reliable = same_label & (conf1 > conf_thresh) & (conf2 > conf_thresh) & \
               (ent1 < ent_thresh) & (ent2 < ent_thresh) & (disagree < disagree_thresh)

    ambiguous = (disagree > amb_disagree_thresh) | (ent1 > amb_ent_thresh) | (ent2 > amb_ent_thresh)
    ambiguous = ambiguous & (~reliable)

    feat1_flat = flatten_patch_map(feat1_p)
    feat2_flat = flatten_patch_map(feat2_p)
    label1_flat = flatten_scalar_map(label1)
    label2_flat = flatten_scalar_map(label2)
    reliable_flat = flatten_scalar_map(reliable)
    ambiguous_flat = flatten_scalar_map(ambiguous)

    conf_mean = flatten_scalar_map((conf1 + conf2) * 0.5)
    ent_mean = flatten_scalar_map((ent1 + ent2) * 0.5)
    dis_rel = flatten_scalar_map(torch.exp(-disagree))
    reliability_score = (conf_mean * (1.0 - ent_mean).clamp(min=0.0) * dis_rel).clamp(min=1e-6)

    return {
        'feat1_flat': feat1_flat,
        'feat2_flat': feat2_flat,
        'label1_flat': label1_flat,
        'label2_flat': label2_flat,
        'reliable_flat': reliable_flat,
        'ambiguous_flat': ambiguous_flat,
        'reliability_score': reliability_score,
    }


def directional_uncertainty_rank_loss(anchor_feat, paired_feat, paired_labels, ambiguous_feat,
                                      num_classes, reliable_mask, max_samples_per_class,
                                      margin_pos_mid, margin_mid_neg):
    anchor_labels = paired_labels
    anchor_idx = sample_balanced_indices(reliable_mask, anchor_labels, max_samples_per_class, num_classes)

    if anchor_idx.numel() == 0 or ambiguous_feat.numel() == 0:
        return anchor_feat.sum() * 0.0, 0

    anchor = F.normalize(anchor_feat[anchor_idx], dim=1)
    positive = F.normalize(paired_feat[anchor_idx], dim=1)
    anchor_label = paired_labels[anchor_idx]
    ambiguous = F.normalize(ambiguous_feat, dim=1)

    s_ap = (anchor * positive).sum(dim=1)
    sim_au = torch.matmul(anchor, ambiguous.t())
    s_au = sim_au.max(dim=1)[0]

    valid_losses = []
    valid_count = 0

    for c in range(num_classes):
        cls_mask = anchor_label == c
        if cls_mask.sum().item() == 0:
            continue

        neg_pool = torch.where(reliable_mask & (paired_labels != c))[0]
        if neg_pool.numel() == 0:
            continue

        neg_feat = F.normalize(paired_feat[neg_pool], dim=1)
        sim_an = torch.matmul(anchor[cls_mask], neg_feat.t())
        s_an = sim_an.max(dim=1)[0]

        loss_rank = F.relu(margin_pos_mid - s_ap[cls_mask] + s_au[cls_mask]) + \
                    F.relu(margin_mid_neg - s_au[cls_mask] + s_an)

        valid_losses.append(loss_rank)
        valid_count += int(loss_rank.numel())

    if not valid_losses:
        return anchor_feat.sum() * 0.0, 0

    return torch.cat(valid_losses, dim=0).mean(), valid_count


def uncertainty_rank_contrastive_loss(feat1, feat2, prob1, prob2, num_classes, patch_size,
                                      conf_thresh, ent_thresh, disagree_thresh,
                                      amb_ent_thresh, amb_disagree_thresh,
                                      margin_pos_mid, margin_mid_neg,
                                      max_samples_per_class):
    group = build_patch_group(
        feat1, feat2, prob1, prob2, patch_size,
        conf_thresh, ent_thresh, disagree_thresh,
        amb_ent_thresh, amb_disagree_thresh
    )

    feat1_flat = group['feat1_flat']
    feat2_flat = group['feat2_flat']
    label1_flat = group['label1_flat']
    label2_flat = group['label2_flat']
    reliable_flat = group['reliable_flat']
    ambiguous_flat = group['ambiguous_flat']

    if feat1_flat.numel() == 0:
        zero = feat1.sum() * 0.0
        return zero, {'anchors': 0, 'reliable': 0, 'ambiguous': 0, 'valid_triplets': 0}, group

    ambiguous_feat1 = feat1_flat[ambiguous_flat]
    ambiguous_feat2 = feat2_flat[ambiguous_flat]

    loss12, valid12 = directional_uncertainty_rank_loss(
        feat1_flat, feat2_flat, label1_flat, ambiguous_feat2,
        num_classes, reliable_flat, max_samples_per_class,
        margin_pos_mid, margin_mid_neg)

    loss21, valid21 = directional_uncertainty_rank_loss(
        feat2_flat, feat1_flat, label2_flat, ambiguous_feat1,
        num_classes, reliable_flat, max_samples_per_class,
        margin_pos_mid, margin_mid_neg)

    valid_total = valid12 + valid21

    if valid_total == 0:
        zero = feat1.sum() * 0.0
        return zero, {
            'anchors': int(reliable_flat.sum().item()),
            'reliable': int(reliable_flat.sum().item()),
            'ambiguous': int(ambiguous_flat.sum().item()),
            'valid_triplets': 0,
        }, group

    loss = 0.5 * (loss12 + loss21)

    return loss, {
        'anchors': int(reliable_flat.sum().item()),
        'reliable': int(reliable_flat.sum().item()),
        'ambiguous': int(ambiguous_flat.sum().item()),
        'valid_triplets': int(valid_total),
    }, group


def reliable_center_compact_loss_from_group(group, num_classes):
    feat1_flat = group['feat1_flat']
    feat2_flat = group['feat2_flat']
    label_flat = group['label1_flat']
    reliable_flat = group['reliable_flat']
    rel_score = group['reliability_score']

    if reliable_flat.sum().item() == 0:
        return feat1_flat.sum() * 0.0, {'compact_pairs': 0, 'compact_classes': 0}

    total_loss = feat1_flat.sum() * 0.0
    used_classes = 0
    used_pairs = 0

    for c in range(num_classes):
        idx = torch.where(reliable_flat & (label_flat == c))[0]
        if idx.numel() < 2:
            continue

        z1 = F.normalize(feat1_flat[idx], dim=1)
        z2 = F.normalize(feat2_flat[idx], dim=1)

        w = rel_score[idx]
        w = (w / (w.sum() + 1e-6)).detach()

        center = (w.unsqueeze(1) * (z1 + z2) * 0.5).sum(dim=0, keepdim=True)
        center = F.normalize(center, dim=1)

        cos1 = (z1 * center).sum(dim=1)
        cos2 = (z2 * center).sum(dim=1)

        cls_loss = ((1.0 - cos1) * rel_score[idx]).sum() / (rel_score[idx].sum() + 1e-6)
        cls_loss += ((1.0 - cos2) * rel_score[idx]).sum() / (rel_score[idx].sum() + 1e-6)
        cls_loss = 0.5 * cls_loss

        total_loss = total_loss + cls_loss
        used_classes += 1
        used_pairs += int(idx.numel())

    if used_classes == 0:
        return feat1_flat.sum() * 0.0, {'compact_pairs': 0, 'compact_classes': 0}

    return total_loss / used_classes, {'compact_pairs': used_pairs, 'compact_classes': used_classes}


def coarse_bool_mask(mask, block_size, threshold):
    """
    mask: [B, H, W], bool
    return: [B, H, W], bool

    Convert pixel-level uncertainty masks into blocks to avoid fragmented exchange.
    """
    if block_size <= 1:
        return mask.bool()

    b, h, w = mask.shape
    x = mask.float().unsqueeze(1)

    k = min(block_size, h, w)
    pooled = F.avg_pool2d(x, kernel_size=k, stride=k)
    coarse = pooled > threshold

    up = F.interpolate(coarse.float(), size=(h, w), mode='nearest')
    return up.squeeze(1).bool()


def build_pixel_reliability_maps(prob1,
                                 prob2,
                                 conf_thresh,
                                 ent_thresh,
                                 disagree_thresh,
                                 amb_ent_thresh,
                                 amb_disagree_thresh,
                                 low_conf_thresh,
                                 unstable_ent_thresh,
                                 unstable_disagree_thresh):
    """
    prob1, prob2: [B, C, H, W]
    return:
    reliable: branch agreement, high confidence, low entropy, and low disagreement;
        these regions can serve as sources.
    ambiguous: high-entropy or high-disagreement regions used as partial targets.
    unstable: class disagreement, low confidence, high entropy, or high disagreement;
        these regions are prioritized as replacement targets.
    """
    with torch.no_grad():
        ent1 = normalized_entropy(prob1)
        ent2 = normalized_entropy(prob2)

        conf1, label1 = torch.max(prob1, dim=1)
        conf2, label2 = torch.max(prob2, dim=1)

        disagree = pixel_symmetric_kl(prob1, prob2)

        same_label = label1 == label2

        reliable = same_label & \
                   (conf1 > conf_thresh) & \
                   (conf2 > conf_thresh) & \
                   (ent1 < ent_thresh) & \
                   (ent2 < ent_thresh) & \
                   (disagree < disagree_thresh)

        ambiguous = ((ent1 > amb_ent_thresh) |
                     (ent2 > amb_ent_thresh) |
                     (disagree > amb_disagree_thresh)) & (~reliable)

        unstable = ((~same_label) |
                    (conf1 < low_conf_thresh) |
                    (conf2 < low_conf_thresh) |
                    (ent1 > unstable_ent_thresh) |
                    (ent2 > unstable_ent_thresh) |
                    (disagree > unstable_disagree_thresh)) & (~reliable)

        conf_mean = 0.5 * (conf1 + conf2)
        ent_mean = 0.5 * (ent1 + ent2)
        reliability_score = (
            conf_mean *
            (1.0 - ent_mean).clamp(min=0.0) *
            torch.exp(-disagree)
        ).clamp(min=1e-6)

    return {
        'reliable': reliable.detach(),
        'ambiguous': ambiguous.detach(),
        'unstable': unstable.detach(),
        'reliability_score': reliability_score.detach(),
        'label1': label1.detach(),
        'label2': label2.detach(),
        'conf1': conf1.detach(),
        'conf2': conf2.detach(),
        'ent1': ent1.detach(),
        'ent2': ent2.detach(),
        'disagree': disagree.detach(),
    }


def build_ure_mixed_sample(x_u,
                           prob1_u,
                           prob2_u,
                           conf_thresh,
                           ent_thresh,
                           disagree_thresh,
                           amb_ent_thresh,
                           amb_disagree_thresh,
                           low_conf_thresh,
                           unstable_ent_thresh,
                           unstable_disagree_thresh,
                           amb_replace_ratio,
                           block_size,
                           reliable_ratio,
                           target_ratio,
                           min_pixels):
    """
    URE: Uncertainty-guided Region Exchange.

    Replace unstable regions with reliable regions from shuffled samples in the batch,
    and partially replace ambiguous regions.

    x_mix:
        Unstable or ambiguous regions replaced by reliable content from another sample.
    pseudo1_mix:
        Model 1 pseudo-label for the mixed image.
    pseudo2_mix:
        Model 2 pseudo-label for the mixed image.
    """
    b = x_u.shape[0]
    device = x_u.device

    if b < 2:
        return None

    maps = build_pixel_reliability_maps(
        prob1=prob1_u,
        prob2=prob2_u,
        conf_thresh=conf_thresh,
        ent_thresh=ent_thresh,
        disagree_thresh=disagree_thresh,
        amb_ent_thresh=amb_ent_thresh,
        amb_disagree_thresh=amb_disagree_thresh,
        low_conf_thresh=low_conf_thresh,
        unstable_ent_thresh=unstable_ent_thresh,
        unstable_disagree_thresh=unstable_disagree_thresh
    )

    reliable = maps['reliable']
    ambiguous = maps['ambiguous']
    unstable = maps['unstable']

    reliable_block = coarse_bool_mask(reliable, block_size=block_size, threshold=reliable_ratio)
    ambiguous_block = coarse_bool_mask(ambiguous, block_size=block_size, threshold=target_ratio)
    unstable_block = coarse_bool_mask(unstable, block_size=block_size, threshold=target_ratio)

    perm = torch.randperm(b, device=device)
    x_src = x_u[perm]

    reliable_src = reliable_block[perm]

    rand_mask = torch.rand_like(ambiguous_block.float()) < amb_replace_ratio
    target_replace = unstable_block | (ambiguous_block & rand_mask)

    exchange_mask = target_replace & reliable_src

    if exchange_mask.float().sum().item() <= min_pixels:
        return None

    m = exchange_mask.float().unsqueeze(1)

    pseudo1 = torch.argmax(prob1_u.detach(), dim=1)
    pseudo2 = torch.argmax(prob2_u.detach(), dim=1)

    pseudo1_src = pseudo1[perm]
    pseudo2_src = pseudo2[perm]

    x_mix = x_u * (1.0 - m) + x_src * m

    pseudo1_mix = torch.where(exchange_mask, pseudo1_src, pseudo1)
    pseudo2_mix = torch.where(exchange_mask, pseudo2_src, pseudo2)

    stats = {
        'ure_exchange_pixels': float(exchange_mask.float().sum().detach().item()),
        'ure_reliable_pixels': float(reliable.float().sum().detach().item()),
        'ure_ambiguous_pixels': float(ambiguous.float().sum().detach().item()),
        'ure_unstable_pixels': float(unstable.float().sum().detach().item()),
        'ure_target_pixels': float(target_replace.float().sum().detach().item()),
    }

    return x_mix, pseudo1_mix.long(), pseudo2_mix.long(), stats


def ure_mixed_loss(model1,
                   model2,
                   x_mix,
                   pseudo1_mix,
                   pseudo2_mix,
                   ce_loss,
                   dice_loss,
                   num_classes,
                   use_dice=True,
                   dice_weight=0.5):
    """
    Cross pseudo supervision on the mixed image: model 1 learns from model 2's
    mixed pseudo-label, and model 2 learns from model 1's mixed pseudo-label.
    """
    outputs1_mix = unwrap_output(model1(x_mix))
    outputs2_mix = unwrap_output(model2(x_mix))

    outputs_soft1_mix = torch.softmax(outputs1_mix, dim=1)
    outputs_soft2_mix = torch.softmax(outputs2_mix, dim=1)

    loss1_ce = ce_loss(outputs1_mix, pseudo2_mix)
    loss2_ce = ce_loss(outputs2_mix, pseudo1_mix)

    if use_dice and num_classes == 2:
        loss1_dice = dice_loss(outputs_soft1_mix, pseudo2_mix.unsqueeze(1))
        loss2_dice = dice_loss(outputs_soft2_mix, pseudo1_mix.unsqueeze(1))
        loss1 = loss1_ce + dice_weight * loss1_dice
        loss2 = loss2_ce + dice_weight * loss2_dice
    else:
        loss1_dice = outputs1_mix.sum() * 0.0
        loss2_dice = outputs2_mix.sum() * 0.0
        loss1 = loss1_ce
        loss2 = loss2_ce

    loss = 0.5 * (loss1 + loss2)

    stats = {
        'ure_loss1_ce': float(loss1_ce.detach().item()),
        'ure_loss2_ce': float(loss2_ce.detach().item()),
        'ure_loss1_dice': float(loss1_dice.detach().item()),
        'ure_loss2_dice': float(loss2_dice.detach().item()),
    }

    return loss, stats


def train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model():
        model = net_factory(
            net_type=args.model,
            in_chns=1,
            class_num=num_classes,
        ).cuda()
        return model

    model1 = create_model()
    model2 = create_model()

    hook1 = PenultimateFeatureHook(model1, num_classes)
    hook2 = PenultimateFeatureHook(model2, num_classes)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(
        base_dir=args.root_path,
        split="train",
        num=None,
        transform=transforms.Compose([RandomGenerator(args.patch_size)])
    )

    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    labeled_len = int(len(db_train) * args.labeled_num)
    unlabeled_len = len(db_train) - labeled_len

    labeled_train_set, unlabeled_train_set = random_split(
        db_train,
        [labeled_len, unlabeled_len],
        generator=torch.Generator().manual_seed(args.seed)
    )

    labeled_idxs = labeled_train_set.indices
    unlabeled_idxs = unlabeled_train_set.indices

    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs,
        unlabeled_idxs,
        batch_size,
        batch_size - args.labeled_bs
    )

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    model1.train()
    model2.train()

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer1 = optim.SGD(model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1

    best_performance1 = -1.0
    best_performance2 = -1.0

    iterator = tqdm(range(max_epoch), ncols=70)

    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            if iter_num == 0:
                logging.info('batch image shape %s label shape %s', tuple(volume_batch.shape), tuple(label_batch.shape))

            outputs1, feat1 = hook1.forward(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            outputs2, feat2 = hook2.forward(volume_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            sup_loss1 = 0.5 * (
                ce_loss(outputs1[:args.labeled_bs], label_batch[:args.labeled_bs].long()) +
                dice_loss(outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            )

            sup_loss2 = 0.5 * (
                ce_loss(outputs2[:args.labeled_bs], label_batch[:args.labeled_bs].long()) +
                dice_loss(outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            )

            pseudo_supervision1 = outputs1.sum() * 0.0
            pseudo_supervision2 = outputs2.sum() * 0.0

            urc_loss = outputs1.sum() * 0.0
            compact_loss = outputs1.sum() * 0.0
            ure_loss = outputs1.sum() * 0.0

            urc_stats = {
                'anchors': 0,
                'reliable': 0,
                'ambiguous': 0,
                'valid_triplets': 0
            }

            compact_stats = {
                'compact_pairs': 0,
                'compact_classes': 0
            }

            ure_stats = {
                'ure_exchange_pixels': 0.0,
                'ure_reliable_pixels': 0.0,
                'ure_ambiguous_pixels': 0.0,
                'ure_unstable_pixels': 0.0,
                'ure_target_pixels': 0.0,
                'ure_loss1_ce': 0.0,
                'ure_loss2_ce': 0.0,
                'ure_loss1_dice': 0.0,
                'ure_loss2_dice': 0.0,
            }

            if args.labeled_bs < batch_size:
                outputs1_u = outputs1[args.labeled_bs:]
                outputs2_u = outputs2[args.labeled_bs:]
                outputs_soft1_u = outputs_soft1[args.labeled_bs:]
                outputs_soft2_u = outputs_soft2[args.labeled_bs:]
                feat1_u = feat1[args.labeled_bs:]
                feat2_u = feat2[args.labeled_bs:]
                volume_u = volume_batch[args.labeled_bs:]

                # -----------------------------
                # Traditional CPS
                # -----------------------------
                pseudo_outputs1 = torch.argmax(outputs_soft1_u.detach(), dim=1)
                pseudo_outputs2 = torch.argmax(outputs_soft2_u.detach(), dim=1)

                pseudo_supervision1 = ce_loss(outputs1_u, pseudo_outputs2)
                pseudo_supervision2 = ce_loss(outputs2_u, pseudo_outputs1)

                # -----------------------------
                # URC
                # -----------------------------
                if args.urc_weight > 0.0 and iter_num >= args.urc_start_iter:
                    urc_loss, urc_stats, urc_group = uncertainty_rank_contrastive_loss(
                        feat1=feat1_u,
                        feat2=feat2_u,
                        prob1=outputs_soft1_u.detach(),
                        prob2=outputs_soft2_u.detach(),
                        num_classes=num_classes,
                        patch_size=args.urc_patch_size,
                        conf_thresh=args.urc_conf_thresh,
                        ent_thresh=args.urc_entropy_thresh,
                        disagree_thresh=args.urc_disagree_thresh,
                        amb_ent_thresh=args.urc_amb_entropy_thresh,
                        amb_disagree_thresh=args.urc_amb_disagree_thresh,
                        margin_pos_mid=args.urc_margin_pos_mid,
                        margin_mid_neg=args.urc_margin_mid_neg,
                        max_samples_per_class=args.urc_max_samples_per_class,
                    )
                else:
                    urc_group = None

                # -----------------------------
                # Compactness
                # -----------------------------
                if args.urc_compact_weight > 0.0 and iter_num >= args.urc_compact_start_iter and urc_group is not None:
                    compact_loss, compact_stats = reliable_center_compact_loss_from_group(
                        urc_group,
                        num_classes
                    )

                # -----------------------------
                # URE: Uncertainty-guided Region Exchange
                # -----------------------------
                if args.ure_weight > 0.0 and iter_num >= args.ure_start_iter:
                    mix_pack = build_ure_mixed_sample(
                        x_u=volume_u.detach(),
                        prob1_u=outputs_soft1_u.detach(),
                        prob2_u=outputs_soft2_u.detach(),
                        conf_thresh=args.ure_conf_thresh,
                        ent_thresh=args.ure_entropy_thresh,
                        disagree_thresh=args.ure_disagree_thresh,
                        amb_ent_thresh=args.ure_amb_entropy_thresh,
                        amb_disagree_thresh=args.ure_amb_disagree_thresh,
                        low_conf_thresh=args.ure_low_conf_thresh,
                        unstable_ent_thresh=args.ure_unstable_entropy_thresh,
                        unstable_disagree_thresh=args.ure_unstable_disagree_thresh,
                        amb_replace_ratio=args.ure_amb_replace_ratio,
                        block_size=args.ure_block_size,
                        reliable_ratio=args.ure_reliable_ratio,
                        target_ratio=args.ure_target_ratio,
                        min_pixels=args.ure_min_pixels
                    )

                    if mix_pack is not None:
                        x_mix, pseudo1_mix, pseudo2_mix, ure_region_stats = mix_pack

                        ure_loss, ure_loss_stats = ure_mixed_loss(
                            model1=model1,
                            model2=model2,
                            x_mix=x_mix,
                            pseudo1_mix=pseudo1_mix,
                            pseudo2_mix=pseudo2_mix,
                            ce_loss=ce_loss,
                            dice_loss=dice_loss,
                            num_classes=num_classes,
                            use_dice=bool(args.ure_use_dice),
                            dice_weight=args.ure_dice_weight
                        )

                        ure_stats.update(ure_region_stats)
                        ure_stats.update(ure_loss_stats)

            model1_loss = sup_loss1 + consistency_weight * pseudo_supervision1 + args.urc_weight * urc_loss
            model2_loss = sup_loss2 + consistency_weight * pseudo_supervision2 + args.urc_weight * urc_loss

            total_loss = model1_loss + model2_loss + \
                         args.urc_compact_weight * compact_loss + \
                         args.ure_weight * ure_loss

            if not torch.isfinite(total_loss):
                raise FloatingPointError('non-finite total_loss at iteration {}'.format(iter_num + 1))

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            total_loss.backward()
            optimizer1.step()
            optimizer2.step()

            iter_num += 1

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9

            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_

            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('consistency_weight/consistency_weight', consistency_weight, iter_num)

            writer.add_scalar('loss/total_loss', total_loss.item(), iter_num)
            writer.add_scalar('loss/model1_loss', model1_loss.item(), iter_num)
            writer.add_scalar('loss/model2_loss', model2_loss.item(), iter_num)
            writer.add_scalar('loss/sup_loss1', sup_loss1.item(), iter_num)
            writer.add_scalar('loss/sup_loss2', sup_loss2.item(), iter_num)
            writer.add_scalar('loss/pseudo_supervision1', pseudo_supervision1.item(), iter_num)
            writer.add_scalar('loss/pseudo_supervision2', pseudo_supervision2.item(), iter_num)
            writer.add_scalar('loss/urc_loss', urc_loss.item(), iter_num)
            writer.add_scalar('loss/compact_loss', compact_loss.item(), iter_num)
            writer.add_scalar('loss/ure_loss', ure_loss.item(), iter_num)

            writer.add_scalar('urc/reliable_patches', urc_stats['reliable'], iter_num)
            writer.add_scalar('urc/ambiguous_patches', urc_stats['ambiguous'], iter_num)
            writer.add_scalar('urc/valid_triplets', urc_stats['valid_triplets'], iter_num)
            writer.add_scalar('urc/compact_pairs', compact_stats['compact_pairs'], iter_num)
            writer.add_scalar('urc/compact_classes', compact_stats['compact_classes'], iter_num)

            writer.add_scalar('ure/exchange_pixels', ure_stats['ure_exchange_pixels'], iter_num)
            writer.add_scalar('ure/reliable_pixels', ure_stats['ure_reliable_pixels'], iter_num)
            writer.add_scalar('ure/ambiguous_pixels', ure_stats['ure_ambiguous_pixels'], iter_num)
            writer.add_scalar('ure/unstable_pixels', ure_stats['ure_unstable_pixels'], iter_num)
            writer.add_scalar('ure/target_pixels', ure_stats['ure_target_pixels'], iter_num)
            writer.add_scalar('ure/loss1_ce', ure_stats['ure_loss1_ce'], iter_num)
            writer.add_scalar('ure/loss2_ce', ure_stats['ure_loss2_ce'], iter_num)
            writer.add_scalar('ure/loss1_dice', ure_stats['ure_loss1_dice'], iter_num)
            writer.add_scalar('ure/loss2_dice', ure_stats['ure_loss2_dice'], iter_num)

            logging.info(
                'iteration %d : total %.4f | model1 %.4f model2 %.4f | '
                'cps1 %.4f cps2 %.4f | urc %.4f compact %.4f ure %.4f | '
                'reliable %d ambiguous %d triplets %d | exchange_pix %.1f unstable_pix %.1f' %
                (
                    iter_num,
                    total_loss.item(),
                    model1_loss.item(),
                    model2_loss.item(),
                    pseudo_supervision1.item(),
                    pseudo_supervision2.item(),
                    float(urc_loss.item()),
                    float(compact_loss.item()),
                    float(ure_loss.item()),
                    urc_stats['reliable'],
                    urc_stats['ambiguous'],
                    urc_stats['valid_triplets'],
                    ure_stats['ure_exchange_pixels'],
                    ure_stats['ure_unstable_pixels'],
                )
            )

            if iter_num % 50 == 0 and volume_batch.shape[0] > 1:
                image = volume_batch[1, 0:1, :, :]
                writer.add_image('train/Image', image, iter_num)

                pred1 = torch.argmax(torch.softmax(outputs1, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model1_Prediction', pred1[1, ...] * 50, iter_num)

                pred2 = torch.argmax(torch.softmax(outputs2, dim=1), dim=1, keepdim=True)
                writer.add_image('train/model2_Prediction', pred2[1, ...] * 50, iter_num)

                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model1.eval()
                metric_list = 0.0

                for _, sampled_val_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_val_batch["image"].unsqueeze(1),
                        sampled_val_batch["label"].unsqueeze(1),
                        model1,
                        classes=num_classes
                    )
                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)

                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        'info/model1_val_{}_dice'.format(class_i + 1),
                        metric_list[class_i, 0],
                        iter_num
                    )
                    writer.add_scalar(
                        'info/model1_val_{}_hd95'.format(class_i + 1),
                        metric_list[class_i, 1],
                        iter_num
                    )

                performance1 = np.mean(metric_list, axis=0)[0]
                mean_hd951 = np.mean(metric_list, axis=0)[1]

                writer.add_scalar('info/model1_val_mean_dice', performance1, iter_num)
                writer.add_scalar('info/model1_val_mean_hd95', mean_hd951, iter_num)

                if performance1 > best_performance1:
                    best_performance1 = performance1

                    save_mode_path = os.path.join(
                        snapshot_path,
                        'model1_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance1, 4))
                    )
                    save_best = os.path.join(
                        snapshot_path,
                        '{}_best_model1.pth'.format(args.model)
                    )

                    torch.save(model1.state_dict(), save_mode_path)
                    torch.save(model1.state_dict(), save_best)

                logging.info(
                    'iteration %d : model1_mean_dice : %f model1_mean_hd95 : %f' %
                    (iter_num, performance1, mean_hd951)
                )

                model1.train()

                model2.eval()
                metric_list = 0.0

                for _, sampled_val_batch in enumerate(valloader):
                    metric_i = test_single_volume(
                        sampled_val_batch["image"].unsqueeze(1),
                        sampled_val_batch["label"].unsqueeze(1),
                        model2,
                        classes=num_classes
                    )
                    metric_list += np.array(metric_i)

                metric_list = metric_list / len(db_val)

                for class_i in range(num_classes - 1):
                    writer.add_scalar(
                        'info/model2_val_{}_dice'.format(class_i + 1),
                        metric_list[class_i, 0],
                        iter_num
                    )
                    writer.add_scalar(
                        'info/model2_val_{}_hd95'.format(class_i + 1),
                        metric_list[class_i, 1],
                        iter_num
                    )

                performance2 = np.mean(metric_list, axis=0)[0]
                mean_hd952 = np.mean(metric_list, axis=0)[1]

                writer.add_scalar('info/model2_val_mean_dice', performance2, iter_num)
                writer.add_scalar('info/model2_val_mean_hd95', mean_hd952, iter_num)

                if performance2 > best_performance2:
                    best_performance2 = performance2

                    save_mode_path = os.path.join(
                        snapshot_path,
                        'model2_iter_{}_dice_{}.pth'.format(iter_num, round(best_performance2, 4))
                    )
                    save_best = os.path.join(
                        snapshot_path,
                        '{}_best_model2.pth'.format(args.model)
                    )

                    torch.save(model2.state_dict(), save_mode_path)
                    torch.save(model2.state_dict(), save_best)

                logging.info(
                    'iteration %d : model2_mean_dice : %f model2_mean_hd95 : %f' %
                    (iter_num, performance2, mean_hd952)
                )

                model2.train()

            if iter_num >= max_iterations:
                break

            _ = time.time()

        if iter_num >= max_iterations:
            iterator.close()
            break

    hook1.close()
    hook2.close()
    writer.close()


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = os.path.abspath(args.snapshot_path)

    os.makedirs(snapshot_path, exist_ok=True)

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    train(args, snapshot_path)
