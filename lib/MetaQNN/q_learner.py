import os
import time
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.backends.cudnn as cudnn
from lib.MetaQNN.cnn import parse as cnn_parse
import state_enumerator as se
from state_string_utils import StateStringUtils
from lib.Models.network import net
from lib.Training.train import train
from lib.Training.validate import validate
from lib.Training.learning_rate_scheduling import LearningRateScheduler
from lib.Utility.pytorch_modelsize import SizeEstimator
from lib.Utility.utils import GPUMem

class QValues:
    def __init__(self):
        self.q = {}

    def save_to_csv(self, q_csv_path):
        start_layer_type = []
        start_layer_depth = []
        start_filter_depth = []
        start_filter_size = []
        start_stride = []
        start_image_size = []
        start_fc_size = []
        start_terminate = []
        end_layer_type = []
        end_layer_depth = []
        end_filter_depth = []
        end_filter_size = []
        end_stride = []
        end_image_size = []
        end_fc_size = []
        end_terminate = []
        utility = []
        for start_state_list in self.q.keys():
            start_state = se.State(state_list=start_state_list)
            for to_state_ix in range(len(self.q[start_state_list]['actions'])):
                to_state = se.State(state_list=self.q[start_state_list]['actions'][to_state_ix])
                utility.append(self.q[start_state_list]['utilities'][to_state_ix])
                start_layer_type.append(start_state.layer_type)
                start_layer_depth.append(start_state.layer_depth)
                start_filter_depth.append(start_state.filter_depth)
                start_filter_size.append(start_state.filter_size)
                start_stride.append(start_state.stride)
                start_image_size.append(start_state.image_size)
                start_fc_size.append(start_state.fc_size)
                start_terminate.append(start_state.terminate)
                end_layer_type.append(to_state.layer_type)
                end_layer_depth.append(to_state.layer_depth)
                end_filter_depth.append(to_state.filter_depth)
                end_filter_size.append(to_state.filter_size)
                end_stride.append(to_state.stride)
                end_image_size.append(to_state.image_size)
                end_fc_size.append(to_state.fc_size)
                end_terminate.append(to_state.terminate)

        q_csv = pd.DataFrame({'start_layer_type' : start_layer_type,
                              'start_layer_depth' : start_layer_depth,
                              'start_filter_depth' : start_filter_depth,
                              'start_filter_size' : start_filter_size,
                              'start_stride' : start_stride,
                              'start_image_size' : start_image_size,
                              'start_fc_size' : start_fc_size,
                              'start_terminate' : start_terminate,
                              'end_layer_type' : end_layer_type,
                              'end_layer_depth' : end_layer_depth,
                              'end_filter_depth' : end_filter_depth,
                              'end_filter_size' : end_filter_size,
                              'end_stride' : end_stride,
                              'end_image_size' : end_image_size,
                              'end_fc_size' : end_fc_size,
                              'end_terminate' : end_terminate,
                              'utility' : utility})
        q_csv.to_csv(q_csv_path, index=False)

    def load_q_values(self, q_csv_path):
        self.q = {}
        q_csv = pd.read_csv(q_csv_path)
        for row in zip(*[q_csv[col].values.tolist() for col in ['start_layer_type',
                                              'start_layer_depth',
                                              'start_filter_depth',
                                              'start_filter_size',
                                              'start_stride',
                                              'start_image_size',
                                              'start_fc_size',
                                              'start_terminate',
                                              'end_layer_type',
                                              'end_layer_depth',
                                              'end_filter_depth',
                                              'end_filter_size',
                                              'end_stride',
                                              'end_image_size',
                                              'end_fc_size',
                                              'end_terminate',
                                              'utility']]):
            start_state = se.State(layer_type = row[0],
                                   layer_depth = row[1],
                                   filter_depth = row[2],
                                   filter_size = row[3],
                                   stride = row[4],
                                   image_size = row[5],
                                   fc_size = row[6],
                                   terminate = row[7]).as_tuple()
            end_state = se.State(layer_type = row[8],
                                 layer_depth = row[9],
                                 filter_depth = row[10],
                                 filter_size = row[11],
                                 stride = row[12],
                                 image_size = row[13],
                                 fc_size = row[14],
                                 terminate = row[15]).as_tuple()
            utility = row[16]

            if start_state not in self.q:
                self.q[start_state] = {'actions': [end_state], 'utilities': [utility]}
            else:
                self.q[start_state]['actions'].append(end_state)
                self.q[start_state]['utilities'].append(utility)

class QLearner:
    def __init__(self,
                 state_space_parameters, 
                 epsilon,
                 WeightInitializer=None,
                 device=None,
                 args=None,
                 save_path =None,
                 state=None,
                 qstore_enc=None,   
                 qstore_dec = None,
                 replaydict = None,
                 replay_dictionary = pd.DataFrame(columns=['net_enc',
                                                         'net_dec',
                                                         'latent_size',
                                                         'reward',
                                                         'epsilon',
                                                         'train_flag'])):

        self.state_list = []
        self.state_space_parameters = state_space_parameters
        self.args = args
        self.enum = se.StateEnumerator(state_space_parameters, args)    
        self.stringutils = StateStringUtils(state_space_parameters, args)

        self.state = se.State('start', 0, 1, 0, 0, args.patch_size, 0, 0) if not state else state
        self.qstore_enc = QValues() if not qstore_enc else qstore_enc
        self.qstore_dec = QValues() if not qstore_dec else qstore_dec
        if  type(qstore_enc) is not type(None):
            self.qstore_enc.load_q_values(qstore_enc)
            self.qstore_dec.load_q_values(qstore_dec)
            self.replay_dictionary = pd.read_csv(replaydict, index_col=0)
        else:
            self.replay_dictionary = replay_dictionary
        self.epsilon=epsilon 
        self.WeightInitializer = WeightInitializer
        self.device = device
        self.gpu_mem_0 = GPUMem(torch.device('cuda') == self.device)
        self.save_path = save_path
        # TODO: hard-coded arc no. to resume from if epsilon < 1
        self.count = args.continue_ite - 1 

    def generate_net(self, epsilon = None, dataset = None): 
        if epsilon != None:
          self.epsilon = epsilon 
        self._reset_for_new_walk()
        state_list_enc = self._run_agent(0)
        self._reset_for_new_walk()
        state_list_dec = self._run_agent(1)

        net_string_enc = self.stringutils.state_list_to_string(state_list_enc, num_classes=len(dataset.val_loader.dataset.class_to_idx))
        net_string_dec = self.stringutils.state_list_to_string(state_list_dec, num_classes=len(dataset.val_loader.dataset.class_to_idx))

        train_flag = True
        if net_string_enc in self.replay_dictionary['net_enc'].values: 
            temp_df = self.replay_dictionary[self.replay_dictionary['net_enc']==net_string_enc]
            if (net_string_dec in temp_df['net_dec'].values) and (int(state_list_enc[-1].fc_size) in\
                 temp_df[temp_df['net_dec'] == net_string_dec]['latent_size'].values):

                reward = temp_df[(temp_df['net_dec']==net_string_dec) & (temp_df['latent_size']==state_list_enc[-1].fc_size)]['reward'].values[0]
                train_flag = temp_df[(temp_df['net_dec']==net_string_dec) & (temp_df['latent_size']==state_list_enc[-1].fc_size)]['train_flag'].values[0]

                self.replay_dictionary = self.replay_dictionary.append(pd.DataFrame([[net_string_enc, net_string_dec, int(state_list_enc[-1].fc_size), reward,\
                                              self.epsilon, train_flag]], columns=['net_enc', 'net_dec', 'latent_size', 'reward', \
                                              'epsilon', 'train_flag']), ignore_index = True)            
                self.count+=1
                self.replay_dictionary.to_csv(os.path.join(self.save_path,'replayDict' + str(self.count) + '.csv'))
                self.sample_replay_for_update()
                self.qstore_enc.save_to_csv(os.path.join(self.save_path,'qValEnc' + str(self.count) + '.csv'))            
                self.qstore_dec.save_to_csv(os.path.join(self.save_path,'qValDec' + str(self.count) + '.csv'))            

            else:
                reward, train_flag = self.__train_val_net(state_list_enc, state_list_dec, self.state_space_parameters, dataset)
                flag_net_string_present = False
                while reward is None:
                    print('-' * 80)
                    print("arc failed mem check or arc can't be constructed..sampling again!")
                    print('-' * 80)
                    self.__reset_for_new_walk()
                    state_list_enc = self.__run_agent()
                    self.__reset_for_new_walk()
                    state_list_dec = self.run_agent()

                    net_string_enc = self.stringutils.state_list_to_string(state_list_enc,\
                                        num_classes=len(dataset.val_loader.dataset.class_to_idx))
                    net_string_dec = self.stringutils.state_list_to_string(state_list_dec,\
                                        num_classes=len(dataset.val_loader.dataset.class_to_idx))

                    if net_string_enc in self.replay_dictionary['net_enc'].values: 
                        temp_df = self.replay_dictionary[self.replay_dictionary['net_enc']==net_string_enc]
                        if (net_string_dec in temp_df['net_dec'].values) and (int(state_list_enc[-1].fc_size) in\
                            temp_df[temp_df['net_dec'] == net_string_dec]['latent_size'].values):

                            reward = temp_df[(temp_df['net_dec']==net_string_dec) & (temp_df['latent_size']==state_list_enc[-1].fc_size)]['reward'].values[0]
                            train_flag = temp_df[(temp_df['net_dec']==net_string_dec) & (temp_df['latent_size']==state_list_enc[-1].fc_size)]['train_flag'].values[0]

                            self.replay_dictionary = self.replay_dictionary.append(pd.DataFrame([[net_string_enc, net_string_dec, int(state_list_enc[-1].fc_size), reward,\
                                                          self.epsilon, train_flag]], columns=['net_enc', 'net_dec', 'latent_size', 'reward', \
                                                          'epsilon', 'train_flag']), ignore_index = True)            
                            self.count+=1
                            self.replay_dictionary.to_csv(os.path.join(self.save_path,'replayDict' + str(self.count) + '.csv'))
                            self.sample_replay_for_update()
                            self.qstore_enc.save_to_csv(os.path.join(self.save_path,'qValEnc' + str(self.count) + '.csv')) 
                            self.qstore_dec.save_to_csv(os.path.join(self.save_path,'qValDec' + str(self.count) + '.csv'))                             
                            flag_net_string_present = True
                            break     

                    reward, train_flag = self.__train_val_net(state_list_enc, state_list_dec, self.state_space_parameters, dataset)

                if flag_net_string_present == False:
                    self.replay_dictionary = self.replay_dictionary.append(pd.DataFrame([[net_string_enc, net_string_dec, int(state_list_enc[-1].fc_size), reward,\
                                                  self.epsilon, train_flag]], columns=['net_enc', 'net_dec', 'latent_size', 'reward', \
                                                  'epsilon', 'train_flag']), ignore_index = True)
                    self.count += 1
                    self.replay_dictionary.to_csv(os.path.join(self.save_path,'replayDict' + str(self.count) + '.csv'))
                    self.sample_replay_for_update()
                    self.qstore_enc.save_to_csv(os.path.join(self.save_path,'qValEnc' + str(self.count) + '.csv')) 
                    self.qstore_dec.save_to_csv(os.path.join(self.save_path,'qValDec' + str(self.count) + '.csv')) 
        else:
            reward, train_flag = self.__train_val_net(state_list_enc, state_list_dec, self.state_space_parameters, dataset)
            flag_net_string_present = False
            while reward is None:
                print('-' * 80)
                print("arc failed mem check or arc can't be constructed..sampling again!")
                print('-' * 80)
                self.__reset_for_new_walk()
                state_list_enc = self.__run_agent()
                self.__reset_for_new_walk()
                state_list_dec = self.run_agent()

                net_string_enc = self.stringutils.state_list_to_string(state_list_enc,\
                                    num_classes=len(dataset.val_loader.dataset.class_to_idx))
                net_string_dec = self.stringutils.state_list_to_string(state_list_dec,\
                                    num_classes=len(dataset.val_loader.dataset.class_to_idx))

                if net_string_enc in self.replay_dictionary['net_enc'].values: 
                    temp_df = self.replay_dictionary[self.replay_dictionary['net_enc']==net_string_enc]
                    if (net_string_dec in temp_df['net_dec'].values) and (int(state_list_enc[-1].fc_size) in\
                        temp_df[temp_df['net_dec'] == net_string_dec]['latent_size'].values):

                        reward = temp_df[(temp_df['net_dec']==net_string_dec) & (temp_df['latent_size']==state_list_enc[-1].fc_size)]['reward'].values[0]
                        train_flag = temp_df[(temp_df['net_dec']==net_string_dec) & (temp_df['latent_size']==state_list_enc[-1].fc_size)]['train_flag'].values[0]

                        self.replay_dictionary = self.replay_dictionary.append(pd.DataFrame([[net_string_enc, net_string_dec, int(state_list_enc[-1].fc_size), reward,\
                                                      self.epsilon, train_flag]], columns=['net_enc', 'net_dec', 'latent_size', 'reward', \
                                                      'epsilon', 'train_flag']), ignore_index = True)            
                        self.count+=1
                        self.replay_dictionary.to_csv(os.path.join(self.save_path,'replayDict' + str(self.count) + '.csv'))
                        self.sample_replay_for_update()
                        self.qstore_enc.save_to_csv(os.path.join(self.save_path,'qValEnc' + str(self.count) + '.csv')) 
                        self.qstore_dec.save_to_csv(os.path.join(self.save_path,'qValDec' + str(self.count) + '.csv'))                         
                        flag_net_string_present = True
                        break     

                reward, train_flag = self.__train_val_net(state_list_enc, state_list_dec, self.state_space_parameters, dataset)

            if flag_net_string_present == False:
                self.replay_dictionary = self.replay_dictionary.append(pd.DataFrame([[net_string_enc, net_string_dec, int(state_list_enc[-1].fc_size), reward,\
                                              self.epsilon, train_flag]], columns=['net_enc', 'net_dec', 'latent_size', 'reward', \
                                              'epsilon', 'train_flag']), ignore_index = True)
                self.count += 1
                self.replay_dictionary.to_csv(os.path.join(self.save_path,'replayDict' + str(self.count) + '.csv'))
                self.sample_replay_for_update()
                self.qstore_enc.save_to_csv(os.path.join(self.save_path,'qValEnc' + str(self.count) + '.csv')) 
                self.qstore_dec.save_to_csv(os.path.join(self.save_path,'qValDec' + str(self.count) + '.csv')) 
    
        # if train_flag == True:
        print('Reward:{}'.format(reward))

    def __train_val_net(self, state_list_enc, state_list_dec, state_space_parameters, dataset):
        # TODO: for average as reward
        # reward = AverageMeter()
        # TODO: for best reward
        reward = 0.
        net_input, _ = next(iter(dataset.val_loader))

        model = net(state_list_enc, state_list_dec, state_space_parameters, net_input, self.args.batch_norm, self.args.drop_out_drop)

        print(model)
        print('-' * 80)
        print('Latent size: {}'.format(model.latent_size))
        print('-' * 80)
        print ('Estimated total gpu usage of model: {gpu_usage:.4f} GB'.format(gpu_usage = model.gpu_usage))
        model_activations_gpu = model.gpu_usage
        cudnn.benchmark = True
        self.WeightInitializer.init_model(model)
        model = model.to(self.device)
        print('available:{}'.format((self.gpu_mem_0.total_mem - self.gpu_mem_0.total_mem*self.gpu_mem_0.get_mem_util())/1024.))
        print('required per gpu with buffer: {}'.format((3./float(self.args.no_gpus)*model_activations_gpu) + 1))
        print('-' * 80)
        if ((self.gpu_mem_0.total_mem - self.gpu_mem_0.total_mem*self.gpu_mem_0.get_mem_util())/1024.) < ((3./float(self.args.no_gpus)*model_activations_gpu) + 1): 
            del model
            return [None] * 2
        elif not((model.conv_no >0 or model.wrn_bb_no>0 or model.fc_no_enc>0) and (model.convT_no>0 or model.wrnT_bb_no>0 or model.fc_no_dec>0)):
            del model
            return [None] * 2
        if int(self.args.no_gpus)>1:
            model = torch.nn.DataParallel(model)
        criterion = nn.BCELoss(size_average = True).to(self.device)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.args.learning_rate,
                               momentum=self.args.momentum,  weight_decay=self.args.weight_decay)
        lr_scheduler = LearningRateScheduler(self.args.lr_wr_epochs, len(dataset.train_loader.dataset), self.args.batch_size,
                                             self.args.learning_rate, self.args.lr_wr_mul, self.args.lr_wr_min)
        save_path_pictures = os.path.join(self.save_path, str(self.count+1))
        if not os.path.exists(save_path_pictures):
            os.mkdir(save_path_pictures)   
        train_flag = True
        epoch = 0
        while epoch < self.args.epochs:
            train(dataset, model, criterion, epoch, optimizer, lr_scheduler, self.device, self.args)
            loss_inverse = validate(dataset, model, criterion, epoch, self.device, self.args, save_path_pictures)
            reward = max(reward, loss_inverse)
            # TODO: include early stopping criterion
            epoch += 1
        del model, criterion, optimizer, lr_scheduler
        return reward, train_flag

    def _reset_for_new_walk(self):

        self.state_list = []
        self.state = se.State('start', 0, 1, 0, 0, self.args.patch_size, 0, 0)

    def _run_agent(self, flag):
        while self.state.terminate == 0:
            self._transition_q_learning(flag)

        return self.state_list

    def _transition_q_learning(self, flag):
        if flag ==0:
            if self.state.as_tuple() not in self.qstore_enc.q:
                self.enum.enumerate_state(self.state, self.qstore_enc.q)        
            action_values = self.qstore_enc.q[self.state.as_tuple()]
        elif flag == 1:
            if self.state.as_tuple() not in self.qstore_dec.q:
                self.enum.enumerate_state(self.state, self.qstore_dec.q)        
            action_values = self.qstore_dec.q[self.state.as_tuple()]

        if np.random.random() < self.epsilon:
            action = se.State(state_list=action_values['actions'][np.random.randint(len(action_values['actions']))])
        else:
            max_q_value = max(action_values['utilities'])
            max_q_indexes = [i for i in range(len(action_values['actions'])) if action_values['utilities'][i]==max_q_value]
            max_actions = [action_values['actions'][i] for i in max_q_indexes]
            action = se.State(state_list=max_actions[np.random.randint(len(max_actions))])

        self.state = self.enum.state_action_transition(self.state, action)
        self._post_transition_updates()

    def _post_transition_updates(self):
        non_bucketed_state = self.state.copy()
        self.state_list.append(non_bucketed_state)

    def sample_replay_for_update(self):
        net_enc = self.replay_dictionary.iloc[-1]['net_enc']
        net_dec = self.replay_dictionary.iloc[-1]['net_dec']
        latent_size = self.replay_dictionary.iloc[-1]['latent_size']
        reward = self.replay_dictionary.iloc[-1]['reward']
        state_list_enc = self.stringutils.convert_model_string_to_states(cnn_parse('net', net_enc))
        state_list_dec =  self.stringutils.convert_model_string_to_states(cnn_parse('net', net_dec))
        self.update_q_value_sequence(state_list_enc, self.accuracy_to_reward(reward), latent_size, 0)
        self.update_q_value_sequence(state_list_dec, self.accuracy_to_reward(reward), latent_size, 1)
        """ update for replay numnber of times with random picking of architectures """
        for i in range(self.state_space_parameters.replay_number -1):
            net_enc = np.random.choice(self.replay_dictionary['net_enc'])
            net_dec = np.random.choice(self.replay_dictionary[self.replay_dictionary['net_enc'] == net_enc]['net_dec'])
            latent_size = np.random.choice(self.replay_dictionary[(self.replay_dictionary['net_enc'] == net_enc) & (self.replay_dictionary['net_dec'] == net_dec)]['latent_size'])
            reward = self.replay_dictionary[(self.replay_dictionary['net_enc'] == net_enc) & (self.replay_dictionary['net_dec'] == net_dec) & (self.replay_dictionary['latent_size'] == latent_size)]['reward'].values[0]
            state_list_enc = self.stringutils.convert_model_string_to_states(cnn_parse('net', net_enc))
            state_list_dec =  self.stringutils.convert_model_string_to_states(cnn_parse('net', net_dec))
            self.update_q_value_sequence(state_list_enc, self.accuracy_to_reward(reward), latent_size, 0)
            self.update_q_value_sequence(state_list_dec, self.accuracy_to_reward(reward), latent_size, 1)

    def accuracy_to_reward(self, acc):
        return acc

    def update_q_value_sequence(self, states, termination_reward, latent_size, flag):
        states[-1].fc_size = latent_size
        self._update_q_value(states[-2], states[-1], termination_reward, flag)
        for i in reversed(range(len(states) - 2)):
            # TODO: q-learning update (set proper q-learning rate in cmdparser.py)
            # self._update_q_value(states[i], states[i+1], 0, flag)

            # TODO: modified update for shorter search schedules (doesn't use q-learning rate in computation)
            self._update_q_value(states[i], states[i+1], termination_reward, flag)

    def _update_q_value(self, start_state, to_state, reward, flag):
        if flag == 0:
            if start_state.as_tuple() not in self.qstore_enc.q:
                self.enum.enumerate_state(start_state, self.qstore_enc.q)
            if to_state.as_tuple() not in self.qstore_enc.q:
                self.enum.enumerate_state(to_state, self.qstore_enc.q)

            actions = self.qstore_enc.q[start_state.as_tuple()]['actions']
            values = self.qstore_enc.q[start_state.as_tuple()]['utilities']

            max_over_next_states = max(self.qstore_enc.q[to_state.as_tuple()]['utilities']) if to_state.terminate != 1 else 0

            action_between_states = self.enum.transition_to_action(start_state, to_state).as_tuple()

            # TODO: q-learning update (set proper q-learning rate in cmdparser.py)
            # values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
            #                                       self.state_space_parameters.learning_rate * (reward + \
            #                                       self.state_space_parameters.discount_factor * max_over_next_states\
            #                                        - values[actions.index(action_between_states)])

            # TODO: modified update for shorter search schedules (doesn't use q-learning rate in computation)
            values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
                                                        (max(reward, values[actions.index(action_between_states)]) -
                                                         values[actions.index(action_between_states)])

            self.qstore_enc.q[start_state.as_tuple()] = {'actions': actions, 'utilities': values}

        elif flag == 1:
            if start_state.as_tuple() not in self.qstore_dec.q:
                self.enum.enumerate_state(start_state, self.qstore_dec.q)
            if to_state.as_tuple() not in self.qstore_dec.q:
                self.enum.enumerate_state(to_state, self.qstore_dec.q)

            actions = self.qstore_dec.q[start_state.as_tuple()]['actions']
            values = self.qstore_dec.q[start_state.as_tuple()]['utilities']

            max_over_next_states = max(self.qstore_dec.q[to_state.as_tuple()]['utilities']) if to_state.terminate != 1 else 0

            action_between_states = self.enum.transition_to_action(start_state, to_state).as_tuple()

            # TODO: q-learning update (set proper q-learning rate in cmdparser.py)
            # values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
            #                                       self.state_space_parameters.learning_rate * (reward + \
            #                                       self.state_space_parameters.discount_factor * max_over_next_states\
            #                                        - values[actions.index(action_between_states)])

            # TODO: modified update for shorter search schedules (doesn't use q-learning rate in computation)
            values[actions.index(action_between_states)] = values[actions.index(action_between_states)] + \
                                                        (max(reward, values[actions.index(action_between_states)]) -
                                                         values[actions.index(action_between_states)])
            self.qstore_dec.q[start_state.as_tuple()] = {'actions': actions, 'utilities': values}

    




