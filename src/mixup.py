import numpy as np
import torch
from torch.utils.data.dataloader import default_collate


def collate_rep_aug(*args):
    batch_imgs, batch_labels = default_collate(*args)
    if isinstance(batch_imgs, list) and len(batch_imgs) > 1:
        batch_labels = batch_labels.repeat(len(batch_imgs))
        batch_imgs = torch.vstack(batch_imgs)

    return batch_imgs, batch_labels


def one_hot(x, num_classes, on_value=1., off_value=0., device='cuda'):
    x = x.long().view(-1, 1)
    return torch.full((x.size()[0], num_classes), off_value, device=device).scatter_(1, x, on_value)


def mixup_target(target, num_classes, lam=1., smoothing=0.0, device='cuda'):
    off_value = smoothing / num_classes
    on_value = 1. - smoothing + off_value
    y1 = one_hot(target, num_classes, on_value=on_value, off_value=off_value, device=device)
    y2 = one_hot(target.flip(0), num_classes, on_value=on_value, off_value=off_value, device=device)
    return y1 * lam + y2 * (1. - lam)


def rand_bbox(img_shape, lam, margin=0., count=None):
    """ Standard CutMix bounding-box
    Generates a random square bbox based on lambda value. This impl includes
    support for enforcing a border margin as percent of bbox dimensions.
    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def rand_bbox_minmax(img_shape, minmax, count=None):
    """ Min-Max CutMix bounding-box
    Inspired by Darknet cutmix impl, generates a random rectangular bbox
    based on min/max percent values applied to each dimension of the input image.
    Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.
    Args:
        img_shape (tuple): Image shape as tuple
        minmax (tuple or list): Min and max bbox ratios (as percent of image size)
        count (int): Number of bbox to generate
    """
    assert len(minmax) == 2
    img_h, img_w = img_shape[-2:]
    cut_h = np.random.randint(int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
    cut_w = np.random.randint(int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
    yl = np.random.randint(0, img_h - cut_h, size=count)
    xl = np.random.randint(0, img_w - cut_w, size=count)
    yu = yl + cut_h
    xu = xl + cut_w
    return yl, yu, xl, xu


def cutmix_bbox_and_lam(img_shape, lam, ratio_minmax=None, correct_lam=True, count=None):
    """ Generate bbox and apply lambda correction.
    """
    if ratio_minmax is not None:
        yl, yu, xl, xu = rand_bbox_minmax(img_shape, ratio_minmax, count=count)
    else:
        yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam or ratio_minmax is not None:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class Mixup:
    """ Mixup/Cutmix that applies different params to each element or whole batch
    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """
    def __init__(self, mixup_alpha=1., cutmix_alpha=0., cutmix_minmax=None, prob=1.0, switch_prob=0.5,
                 mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def _params_per_elem(self, batch_size):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=np.bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size))
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size, dtype=np.bool)
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(np.random.rand(batch_size) < self.mix_prob, lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    # TODO: readapt this to exploit tensor primitives
    def _mix_elem(self, x1, x2):
        batch_size = len(x1)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x = x1.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x1[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x2[j][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x2[j] * (1 - lam)
        return x, torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    # TODO: readapt this to exploit tensor primitives
    def _mix_pair(self, x1, x2):
        batch_size = len(x1)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x = x1.clone()  # need to keep an unmodified original for mixing source
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                        x1[i].shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
                    x[i][:, yl:yh, xl:xh] = x2[j][:, yl:yh, xl:xh]
                    x[j][:, yl:yh, xl:xh] = x2[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x2[j] * (1 - lam)
                    x[j] = x[j] * lam + x2[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return x, torch.tensor(lam_batch, device=x.device, dtype=x.dtype).unsqueeze(1)

    def _mix_batch(self, x1, x2):
        lam, use_cutmix = self._params_per_batch()
        x = x1.clone()
        if lam == 1.:
            return x, 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = cutmix_bbox_and_lam(
                x1.shape, lam, ratio_minmax=self.cutmix_minmax, correct_lam=self.correct_lam)
            x[:, :, yl:yh, xl:xh] = x2.flip(0)[:, :, yl:yh, xl:xh]
        else:
            x2_flipped = x2.flip(0).mul(1. - lam)
            x.mul_(lam).add_(x2_flipped)
        return x, lam

    def __call__(self, x1, x2):
        # assert len(x1) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            x, lam = self._mix_elem(x1, x2)
        elif self.mode == 'pair':
            x, lam = self._mix_pair(x1, x2)
        else:
            x, lam = self._mix_batch(x1, x2)
        return x, lam


class MultiViewCollateMixup(Mixup):
    """ Collate w/ Mixup/Cutmix for multiview batches
    """

    def __init__(self, mixup_alpha=1, cutmix_alpha=0, cutmix_minmax=None, prob=1, switch_prob=0.5,
                mode='batch', correct_lam=True, label_smoothing=0.1, num_classes=1000, views_to_mix="all"):
        super().__init__(mixup_alpha, cutmix_alpha, cutmix_minmax, prob, switch_prob,
                    mode, correct_lam, label_smoothing, num_classes)
        self.views_to_mix = views_to_mix

    def __call__(self, samples, _=None):

        labels, view_lists = zip(*samples)
        
        # stack views to build the multi_view batch
        n_views = len(view_lists[0])
        multiview_batch = []
        for i in range(n_views):
            multiview_batch.append(torch.stack([views[i] for views in view_lists]))
        
        # stack labels
        labels = torch.tensor(labels)

        assert len(multiview_batch) % 2 == 0

        lambdas = []
        mixed = []
        n_mixed_views = 0
        for (x1, x2) in zip(multiview_batch[::2], multiview_batch[1::2]):
            mix_x1x2, lam = super().__call__(x1, x2)
            mixed.append(mix_x1x2)
            lambdas.append(lam)

            n_mixed_views += 1
            # exit the loop on the first iter if we only want to mix globals
            if self.views_to_mix == "globals":
                break

        # insert mixed views just after the original pair of views
        # multiview_batch will look like (v1, v2, mix_v1v2, v3, v4, mix_v3v4, ...)
        mixup_idxs = []
        for i in range(n_mixed_views * 2, 0, -2):
            multiview_batch.insert(i, mixed.pop((i - 1) // 2))
            mixup_idxs.append(i)

        return labels, multiview_batch, lambdas, mixup_idxs


class CollateMixup(Mixup):
    """ Collate w/ Mixup/Cutmix
    """

    def __init__(self, extend_batch=True, **kwargs):
        super().__init__(**kwargs)
        self.extend_batch = extend_batch

    def __call__(self, samples, _=None):

        batch, labels = collate_rep_aug(samples)

        mixed, lam = super().__call__(batch, batch)
        mixed_labels = mixup_target(labels, self.num_classes, lam, self.label_smoothing, mixed.device)

        if not self.extend_batch:
            return mixed, mixed_labels

        return torch.vstack([batch, mixed]), torch.vstack([one_hot(labels, self.num_classes, device=labels.device), mixed_labels])
