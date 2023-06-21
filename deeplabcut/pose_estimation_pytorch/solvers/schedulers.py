from torch.optim.lr_scheduler import _LRScheduler

class LRListScheduler(_LRScheduler):

    def __init__(self, optimizer, last_epoch=-1, verbose=False, milestones=[10], lr_list=[0.001]):
        self.milestones = milestones
        self.lr_list = lr_list
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [lr for lr in self.lr_list[self.milestones.index(self.last_epoch)]]
        