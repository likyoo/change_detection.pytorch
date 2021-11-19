import warnings

from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.optim.optimizer import Optimizer

__all__ = ['GradualWarmupScheduler', 'PolyScheduler']


class GradualWarmupScheduler(_LRScheduler):
    """https://github.com/ildoonet/pytorch-gradual-warmup-lr
    Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)


class PolyScheduler(_LRScheduler):
    r"""Decays the learning rate of each parameter group using a polynomial LR scheduler.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        power (float): Polynomial factor of learning rate decay.
        total_steps (int): The total number of steps in the cycle. Note that
            if a value is not provided here, then it must be inferred by providing
            a value for epochs and steps_per_epoch.
            Default: None
        epochs (int): The number of epochs to train for. This is used along
            with steps_per_epoch in order to infer the total number of steps in the cycle
            if a value for total_steps is not provided.
            Default: None
        steps_per_epoch (int): The number of steps per epoch to train for. This is
            used along with epochs in order to infer the total number of steps in the
            cycle if a value for total_steps is not provided.
            Default: None
        by_epoch (bool): If ``True``, the learning rate will be updated with the epoch
            and `steps_per_epoch` and `total_steps` will be ignored. If ``False``,
            the learning rate will be updated with the batch, you must define either
            `total_steps` or (`epochs` and `steps_per_epoch`).
            Default: ``False``.
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.
        last_epoch (int): The index of the last batch. This parameter is used when
            resuming a training job. Since `step()` should be invoked after each
            batch instead of after each epoch, this number represents the total
            number of *batches* computed, not the total number of epochs computed.
            When last_epoch=-1, the schedule is started from the beginning.
            Default: -1
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.PolyScheduler(optimizer, power=0.9, steps_per_epoch=len(data_loader), epochs=10)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()

        OR

        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.PolyScheduler(optimizer, power=0.9, epochs=10, by_epoch=True)
        >>> for epoch in range(10):
        >>>     train_epoch(...)
        >>>     scheduler.step()


    https://github.com/likyoo/change_detection.pytorch/blob/main/change_detection_pytorch/utils/lr_scheduler.py
    """

    def __init__(self,
                 optimizer,
                 power=1.0,
                 total_steps=None,
                 epochs=None,
                 steps_per_epoch=None,
                 by_epoch=False,
                 min_lr=0,
                 last_epoch=-1,
                 verbose=False):

        # Validate optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.power = power
        self.by_epoch = by_epoch
        self.min_lr = min_lr

        # Validate total_steps
        if by_epoch:
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError("Expected positive integer epochs, but got {}".format(epochs))
            if steps_per_epoch is not None or total_steps is not None:
                warnings.warn("`steps_per_epoch` and `total_steps` will be ignored if `by_epoch` is True, "
                              "please use `epochs`.", UserWarning)
            self.total_steps = epochs
        elif total_steps is None and epochs is None and steps_per_epoch is None:
            raise ValueError("You must define either total_steps OR (epochs AND steps_per_epoch)")
        elif total_steps is not None:
            if total_steps <= 0 or not isinstance(total_steps, int):
                raise ValueError("Expected positive integer total_steps, but got {}".format(total_steps))
            self.total_steps = total_steps
        else:
            if epochs <= 0 or not isinstance(epochs, int):
                raise ValueError("Expected positive integer epochs, but got {}".format(epochs))
            if steps_per_epoch <= 0 or not isinstance(steps_per_epoch, int):
                raise ValueError("Expected positive integer steps_per_epoch, but got {}".format(steps_per_epoch))
            self.total_steps = epochs * steps_per_epoch

        super(PolyScheduler, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        step_num = self.last_epoch

        if step_num > self.total_steps:
            raise ValueError("Tried to step {} times. The specified number of total steps is {}"
                             .format(step_num + 1, self.total_steps))

        if step_num == 0:
            return self.base_lrs

        coeff = (1 - step_num / self.total_steps) ** self.power

        return [(base_lr - self.min_lr) * coeff + self.min_lr
                for base_lr in self.base_lrs]



if __name__ == '__main__':
    # https://github.com/ildoonet/pytorch-gradual-warmup-lr
    # import torch
    # from torch.optim.lr_scheduler import StepLR
    # from torch.optim.sgd import SGD
    #
    # model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    # optim = SGD(model, 0.1)
    #
    # # scheduler_warmup is chained with schduler_steplr
    # scheduler_steplr = StepLR(optim, step_size=10, gamma=0.1)
    # scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)
    #
    # # this zero gradient update is needed to avoid a warning message, issue #8.
    # optim.zero_grad()
    # optim.step()
    #
    # for epoch in range(1, 20):
    #     scheduler_warmup.step(epoch)
    #     print(epoch, optim.param_groups[0]['lr'])
    #
    #     optim.step()    # backward pass (update network)

    import matplotlib.pyplot as plt
    import torch

    EPOCH = 10
    LEN_DATA = 10

    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = torch.optim.SGD(params=model, lr=0.1)
    scheduler = PolyScheduler(optimizer, power=0.9, min_lr=1e-4, epochs=EPOCH, steps_per_epoch=LEN_DATA, by_epoch=False)
    plt.figure()

    x = list(range(EPOCH*LEN_DATA))
    y = []
    for epoch in range(EPOCH):
        for batch in range(LEN_DATA):
            print(epoch, 'lr={:.6f}'.format(scheduler.get_last_lr()[0]))
            y.append(scheduler.get_last_lr()[0])
            scheduler.step()

    plt.plot(x, y)
    plt.show()
