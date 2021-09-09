from torch.optim.lr_scheduler import MultiStepLR


class MultiStepLRStr(MultiStepLR):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        super(MultiStepLRStr, self).__init__(optimizer, milestones, gamma=gamma, last_epoch=last_epoch)
        self.gamma = gamma
        self.milestones = milestones

    def __str__(self):
        format_string = self.__class__.__name__ + ' ('
        format_string += '    milestones: {0}\n'.format(self.milestones)
        format_string += '    gamma: {0}\n'.format(self.gamma)
        format_string += ')'
        return format_string
