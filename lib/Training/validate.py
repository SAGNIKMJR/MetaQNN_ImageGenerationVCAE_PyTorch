import time
import torch
import os
from torchvision.utils import save_image
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

def validate(dataset, model, criterion, epoch, device, args, save_path_pictures):
    """
    Trains/updates the model for one epoch on the training dataset.

    Parameters:
        train_loader (torch.utils.data.DataLoader): The trainset dataloader
        model (torch.nn.module): Model to be trained
        criterion (torch.nn.criterion): Loss function
        epoch (int): Continuous epoch counter
        device (str): device name where data is transferred to
        args (dict): Dictionary of (command line) arguments.
            Needs to contain learning_rate (float), momentum (float),
            weight_decay (float), nesterov momentum (bool), lr_dropstep (int),
            lr_dropfactor (float), print_freq (int) and expand (bool).
    """

    elbo_losses = AverageMeter()
    rec_losses = AverageMeter()
    kl_losses = AverageMeter()

    # switch to train mode
    model.eval()

    for i, (input, target) in enumerate(dataset.train_loader):
        input, target = input.to(device), input.to(device)

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

        if i % args.print_freq == 0:
            save_image((input.data).view(-1,input.size(1),input.size(2),input.size(3)), os.path.join(save_path_pictures, 'input_epoch_' + str(epoch) + '_ite_'+str(i+1)+'.png'))            
            save_image((output.data).view(-1,output.size(1),output.size(2),output.size(3)), os.path.join(save_path_pictures, 'epoch_' + str(epoch) + '_ite_'+str(i+1)+'.png'))
            if args.no_gpus>1:
                sample = torch.randn(input.size(0), model.module.latent_size).to(device)
                sample = model.module.sample(sample).cpu()
            else:
                sample = torch.randn(input.size(0), model.latent_size).to(device)
                sample = model.sample(sample).cpu()                
            save_image((sample.view(-1,input.size(1),input.size(2),input.size(3))), os.path.join(save_path_pictures, 'sample_epoch_' + str(epoch) + '_ite_'+str(i+1)+'.png'))
        del output, input, target

    print(' * Train: ELBO Loss {elbo_losses.avg:.3f} Reconstruction Loss {rec_losses.avg:.3f} KL Divergence {kl_losses.avg:.3f}'\
        .format(elbo_losses=elbo_losses, rec_losses=rec_losses, kl_losses=kl_losses))
    print('-' * 80)
    return 1./elbo_losses.avg
