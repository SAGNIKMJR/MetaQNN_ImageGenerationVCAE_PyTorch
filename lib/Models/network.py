import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
from lib.Utility import FeatureOperations as FO

# TODO: bias = False

class WRNBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout=0.0, batchnorm=1e-3):
        super(WRNBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        """ 
            NOTE: batch norm in commented lines
        """
        self.bn1 = nn.BatchNorm2d(in_planes, eps=batchnorm)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        """ 
            NOTE: batch norm in commented lines
        """
        self.bn2 = nn.BatchNorm2d(out_planes, eps=batchnorm)
        self.relu2 = nn.ReLU(inplace=True)

        self.droprate = dropout
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                                                stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        """ 
            NOTE: batch norm in commented lines
        """
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            # x = F.relu(x)

        else:
            out = self.relu1(self.bn1(x))
            # out = F.relu(x)
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        # out = self.relu2(self.conv1(out if self.equalInOut else x))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)

        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class net(nn.Module):

    def __init__(self, state_list_enc, state_list_dec, state_space_parameters, net_input, bn_val, do_drop):
        super(net, self).__init__()
        self.state_list_enc = state_list_enc
        self.state_list_dec = state_list_dec
        self.state_space_parameters = state_space_parameters
        self.batch_size = net_input.size(0)
        self.num_colors = net_input.size(1)
        self.image_size = net_input.size(2)
        self.bn_value = bn_val
        self.do_drop = do_drop
        self.gpu_usage = 32 * self.batch_size * self.num_colors * self.image_size * self.image_size
        feature_list = []	
        classifier_list = []	
        temp_defeature_list = []
        temp_declassifier_list = []  
        defeature_list = []
        declassifier_list = []											
        wrn_bb_no = conv_no = convT_no = wrnT_bb_no = fc_no = relu_no = drop_no = bn_no = 0 
        feature = 1
        out_channel = self.num_colors
        no_feature = self.num_colors*((self.image_size)**2)
        last_image_size = self.image_size
        self.final_no_feature = no_feature
        print('-' * 80)
        print('VCAE')
        print('-' * 80)
        for state_no, state in enumerate(self.state_list_enc):
            if state_no == len(self.state_list_enc)-1:
                break
            if state.layer_type == 'fc':
                feature = 0
            if feature == 1:
                if state.layer_type == 'wrn':
                    wrn_bb_no += 1
                    in_channel = out_channel
                    out_channel = state.filter_depth
                    no_feature = ((state.image_size)**2)*(out_channel)
                    last_image_size = state.image_size
                    # TODO: fix padding, will work for stride = 1 only
                    feature_list.append(('wrn_bb' + str(wrn_bb_no), WRNBasicBlock(in_channel, out_channel,
                                        stride = state.stride, dropout = 0, batchnorm = self.bn_value)))
                    self.gpu_usage += 32*(3*3*in_channel*out_channel + 3*3*out_channel*out_channel + int(in_channel!=out_channel)*in_channel*out_channel)
                    self.gpu_usage += 32*self.batch_size*state.image_size*state.image_size*state.filter_depth*(2 + int(in_channel!=out_channel))

                    # feature_list.append(('dropout' + str(wrn_bb_no), nn.Dropout2d(p = self.do_drop)))   
                elif state.layer_type == 'conv':
                    conv_no += 1
                    in_channel = out_channel
                    out_channel = state.filter_depth
                    no_feature = ((state.image_size)**2)*(out_channel)
                    # TODO: include option for 'SAME'
                    # TODO: fix padding, will work for stride = 1 only
                    if ((last_image_size - state.filter_size)/state.stride + 1)%2 != 0:
                        feature_list.append(('conv' + str(conv_no), nn.Conv2d(in_channel, out_channel,
                                            state.filter_size, stride = state.stride, padding = 1, bias = False))) 
                        last_image_size = (last_image_size - state.filter_size + 2)/state.stride + 1
                    else:
                        feature_list.append(('conv' + str(conv_no), nn.Conv2d(in_channel, out_channel,
                                            state.filter_size, stride = state.stride, padding = 0, bias = False))) 
                        last_image_size = (last_image_size - state.filter_size)/state.stride + 1
                    """
                        NOTE:
                        uncomment to include batch norm
                    """                 
                    bn_no += 1
                    feature_list.append(('batchnorm' + str(bn_no), nn.BatchNorm2d(num_features=out_channel, eps=self.bn_value)))
                    relu_no += 1
                    feature_list.append(('relu' + str(relu_no), nn.ReLU(inplace = True)))
                    # feature_list.append(('dropout' + str(conv_no), nn.Dropout2d(p = do_drop)))
                    self.gpu_usage += 32*(state.image_size * state.image_size * state.filter_depth * self.batch_size \
                                        + in_channel * out_channel * state.filter_size * state.filter_size) 
            else:
                if state.layer_type == 'fc':
                    fc_no += 1
                    in_feature = no_feature
                    no_feature = (state.fc_size)
                    classifier_list.append(('fc' + str(fc_no), nn.Linear(in_feature, no_feature, bias = False)))
                    """
                        NOTE:
                        uncomment to include batch norm
                    """      
                    classifier_list.append(('batchnorm_fc' + str(fc_no), nn.BatchNorm1d(num_features=no_feature, eps=self.bn_value)))
                    classifier_list.append(('relu_fc' + str(fc_no), nn.ReLU(inplace=True)))
                    # classifier_list.append(('dropout' + str(fc_no), nn.Dropout(p = do_drop)))
                    self.gpu_usage += 32 * (no_feature * self.batch_size + in_feature * no_feature)

        self.wrn_bb_no = wrn_bb_no
        self.conv_no = conv_no
        self.fc_no_enc = fc_no
        self.features_list = nn.Sequential(collections.OrderedDict(feature_list))
        self.classifiers_list = nn.Sequential(collections.OrderedDict(classifier_list))
        self.linear2mean = nn.Linear(no_feature, self.state_list_enc[-1].fc_size)
        self.linear2std = nn.Linear(no_feature, self.state_list_enc[-1].fc_size)
        self.latent_size = self.state_list_enc[-1].fc_size

        feature = 1
        out_channel = self.num_colors
        no_feature = self.num_colors*((self.image_size)**2)
        last_image_size = self.image_size
        self.gpu_usage += 32 * self.batch_size * self.num_colors * self.image_size * self.image_size

        for state_no, state in enumerate(self.state_list_dec):
            if state_no == len(self.state_list_dec)-1:
                break
            if state.layer_type == 'fc':
                feature = 0
            if feature == 1:
                if state.layer_type == 'wrn':
                    in_channel = out_channel
                    out_channel = state.filter_depth
                    no_feature = ((state.image_size)**2)*(out_channel)
                    last_image_size = state.image_size
                    # TODO: fix padding, will work for stride = 1 only
                    temp_defeature_list.append((0, out_channel, in_channel, state.filter_size, 0))
                    self.gpu_usage += 32*(3*3*in_channel*out_channel + 3*3*out_channel*out_channel + int(in_channel!=out_channel)*in_channel*out_channel)
                    self.gpu_usage += 32*self.batch_size*state.image_size*state.image_size*state.filter_depth*(2 + int(in_channel!=out_channel))

                    # feature_list.append(('dropout' + str(wrn_bb_no), nn.Dropout2d(p = self.do_drop)))   
                elif state.layer_type == 'conv':
                    in_channel = out_channel
                    out_channel = state.filter_depth
                    no_feature = ((state.image_size)**2)*(out_channel)
                    # TODO: include option for 'SAME'
                    # TODO: fix padding, will work for stride = 1 only
                    if ((last_image_size - state.filter_size)/state.stride + 1)%2 != 0:
                        temp_defeature_list.append((1, out_channel, in_channel, state.filter_size,1)) 
                        last_image_size = (last_image_size - state.filter_size + 2)/state.stride + 1
                    else:
                        temp_defeature_list.append((1, out_channel, in_channel, state.filter_size,0)) 
                        last_image_size = (last_image_size - state.filter_size)/state.stride + 1          
                    self.gpu_usage += 32*(state.image_size * state.image_size * state.filter_depth * self.batch_size \
                                        + in_channel * out_channel * state.filter_size * state.filter_size) 
            else:
                if state.layer_type == 'fc':
                    in_feature = no_feature
                    no_feature = (state.fc_size)
                    temp_declassifier_list.append((no_feature, in_feature))
                    self.gpu_usage += 32 * (no_feature * self.batch_size + in_feature * no_feature)
        # NOTE: features for the latent space counted twice, thus subtracted once
        # self.gpu_usage -= 32 * (no_feature * self.batch_size + in_feature * no_feature)
        self.final_image_size = last_image_size
        self.fc_no_dec = len(temp_declassifier_list)
        for i in range(len(declassifier_list)):
            fc_no += 1
            index = len(temp_declassifier_list) -1 - i
            if i == 0:
                declassifier_list.append(('fc' + str(fc_no), nn.Linear(self.latent_size, temp_declassifier_list[index][1])))
            else:
                declassifier_list.append(('fc' + str(fc_no), nn.Linear(temp_declassifier_list[index][0], temp_declassifier_list[index][1])))
            bn_no += 1
            declassifier_list.append(('batchNorm' + str(bn_no), nn.BatchNorm1d(temp_declassifier_list[index][1])))
            relu_no += 1
            if len(temp_defeature_list) == 0 and i==(len(temp_declassifier_list)-1):
                declassifier_list.append(('relu' + str(relu_no), nn.Sigmoid()))
            else:    
                declassifier_list.append(('relu' + str(relu_no), nn.ReLU(inplace = True)))
        if len(temp_declassifier_list) == 0:
            fc_no += 1
            declassifier_list.append(('fc' + str(fc_no), nn.Linear(self.latent_size, out_channel*(last_image_size**2))))
            bn_no += 1
            declassifier_list.append(('batchNorm' + str(bn_no), nn.BatchNorm1d(out_channel*(last_image_size**2))))            
            relu_no += 1
            declassifier_list.append(('relu' + str(relu_no), nn.ReLU(inplace = True)))
        for i in range(len(temp_defeature_list) - 1):
            index = len(temp_defeature_list) - 1 - i 
            if temp_defeature_list[index][0] == 1: 
                convT_no += 1
                defeature_list.append(('convT' + str(convT_no), nn.ConvTranspose2d(temp_defeature_list[index][1], \
                                        temp_defeature_list[index][2], temp_defeature_list[index][3], stride = 2, padding = temp_defeature_list[index][4])))
                bn_no += 1
                defeature_list.append(('batchNorm' + str(bn_no), nn.BatchNorm2d(temp_defeature_list[index][2])))
                relu_no += 1
                defeature_list.append(('relu' + str(relu_no), nn.ReLU(inplace = True)))
            elif temp_defeature_list[index][0] == 0:
                wrnT_bb_no += 1
                defeature_list.append(('wrnT_bb_' + str(wrnT_bb_no), WRNBasicBlock(temp_defeature_list[index][1], \
                                        temp_defeature_list[index][2], stride = 1)))
        if len(temp_defeature_list)>0:
            convT_no += 1
            defeature_list.append(('convT' + str(convT_no), nn.ConvTranspose2d(temp_defeature_list[0][1], \
                                    temp_defeature_list[0][2], temp_defeature_list[0][3], stride = 2, padding = temp_defeature_list[0][4])))
            defeature_list.append(('sigmoid', nn.Sigmoid()))
        self.convT_no = convT_no
        self.wrnT_bb_no = wrnT_bb_no
        self.declassifiers_list = nn.Sequential(collections.OrderedDict(declassifier_list))
        self.defeatures_list = nn.Sequential(collections.OrderedDict(defeature_list))
        self.gpu_usage /= (8.*1024*1024*1024)

    def forward(self, x):
        # NOTE: double check to see if the VAE will work, a check will be there in the __train_val_net() too
        if (self.conv_no>0 or self.wrn_bb_no>0 or self.fc_no_enc>0) and\
            (self.convT_no>0 or self.wrnT_bb_no>0 or self.fc_no_dec>0): 
            x = self.features_list(x)
            x = self.classifiers_list(x.view(x.size(0), -1))
            mean = self.linear2mean(x)
            std = self.linear2std(x) 
            eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add(mean)
            x = self.declassifiers_list(z)
            x = self.defeatures_list(x.view(x.size(0), -1, self.final_image_size, self.final_image_size))
            x = x.view(x.size(0),-1,self.image_size,self.image_size)
            return x, mean, std
    def sample(self, z):
        x = self.declassifiers_list(z)
        x = self.defeatures_list(x.view(x.size(0), -1, self.final_image_size, self.final_image_size))
        x = x.view(x.size(0),-1,self.image_size,self.image_size)
        return x