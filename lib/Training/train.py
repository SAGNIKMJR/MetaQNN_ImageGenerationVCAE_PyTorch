import time
import torch
from lib.Utility.metrics import AverageMeter
from lib.Utility.metrics import accuracy
def mod_loss(mean_out, std_out, mean, std):
    """ 
        NOTE:
        source: https://github.com/skerit/cmusphinx/blob/master/SphinxTrain/python/cmusphinx/divergence.py
        source2: https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
    """
    qm = mean 
    pm = mean_out
    qv = std**2
    pv = std_out**2
    dpv = pv
    dqv = qv
    iqv = 1./qv
    diff = qm - pm
    kl = 0.5*torch.mean(torch.log(qv/pv+1e-8)+iqv*pv+diff*iqv*diff-1)

def train(dataset, model, criterion, epoch, optimizer, lr_scheduler, device, args):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        train_loader (torch.utils.data.DataLoader): The trainset dataloader
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        optimizer (torch.optim.optimizer): optimizer instance like SGD or Adam
        lr_scheduler (Training.LearningRateScheduler): class implementing learning rate schedules
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain learning_rate (float), momentum (float),
            weight_decay (float), nesterov momentum (bool), lr_dropstep (int),
            lr_dropfactor (float), print_freq (int) and expand (bool).
    """

    batch_time = AverageMeter()
    data_time = AverageMeter()
    elbo_losses = AverageMeter()
    rec_losses = AverageMeter()
    kl_losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(dataset.train_loader):
        input, target = input.to(device), input.to(device)
        # measure data loading time
        data_time.update(time.time() - end)

        # adjust the learning rate if applicable
        lr_scheduler.adjust_learning_rate(optimizer, i + 1)

        # compute output, mean and std
        output, mean, std = model(input)

        # compute loss 
        kl_loss = -0.5*torch.mean(1+torch.log(1e-8+std**2)-(mean**2)-(std**2))
        rec_loss = criterion(output, target)
        elbo_loss = rec_loss + kl_loss

        # record loss
        rec_losses.update(rec_loss.item(), input.size(0))
        kl_losses.update(kl_loss.item(), input.size(0))
        elbo_losses.update(elbo_loss.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        elbo_loss.backward()
        optimizer.step()
        del output, input, target
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'ELBO Loss {elbo_losses.val:.4f} ({elbo_losses.avg:.4f})\t'
                  'Reconstruction Loss {rec_losses.val:.3f} ({rec_losses.avg:.3f})\t'
                  'KL Divergence {kl_losses.val:.3f} ({kl_losses.avg:.3f})\t'.format(
                   epoch, i, len(dataset.train_loader), batch_time=batch_time,
                   data_time=data_time, elbo_losses=elbo_losses, rec_losses=rec_losses, kl_losses=kl_losses))


    lr_scheduler.scheduler_epoch += 1

    print(' * Train: ELBO Loss {elbo_losses.avg:.3f} Reconstruction Loss {rec_losses.avg:.3f} KL Divergence {kl_losses.avg:.3f}'\
        .format(elbo_losses=elbo_losses, rec_losses=rec_losses, kl_losses=kl_losses))
    print('-' * 80)
