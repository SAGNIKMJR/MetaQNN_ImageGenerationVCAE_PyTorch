import math
import numpy as np
from operator import itemgetter
import cnn

class State:
    def __init__(self,
                 layer_type=None,        # String -- conv, pool, fc, softmax
                 layer_depth=None,       # Current depth of network
                 filter_depth=None,      # Used for conv, 0 when not conv
                 filter_size=None,       # Used for conv and pool, 0 otherwise
                 stride=None,            # Used for conv and pool, 0 otherwise
                 image_size=None,        # Used for any layer that maintains square input (conv and pool), 0 otherwise
                 fc_size=None,           # Used for fc and softmax -- number of neurons in layer
                 terminate=None,
                 state_list=None):       # can be constructed from a list instead, list takes precedent
        if not state_list:
            self.layer_type = layer_type
            self.layer_depth = layer_depth
            self.filter_depth = filter_depth
            self.filter_size = filter_size
            self.stride = stride
            self.image_size = image_size
            self.fc_size = fc_size
            self.terminate = terminate
        else:
            self.layer_type = state_list[0]
            self.layer_depth = state_list[1]
            self.filter_depth = state_list[2]
            self.filter_size = state_list[3]
            self.stride = state_list[4]
            self.image_size = state_list[5]
            self.fc_size = state_list[6]
            self.terminate = state_list[7]

    def as_tuple(self):
        return (self.layer_type, 
                self.layer_depth, 
                self.filter_depth, 
                self.filter_size, 
                self.stride, 
                self.image_size,
                self.fc_size,
                self.terminate)
    def as_list(self):
        return list(self.as_tuple())
    def copy(self):
        return State(self.layer_type, 
                     self.layer_depth, 
                     self.filter_depth, 
                     self.filter_size, 
                     self.stride, 
                     self.image_size,
                     self.fc_size,
                     self.terminate)

class StateEnumerator:
    def __init__(self, state_space_parameters, args):
        # Limits
        self.ssp = state_space_parameters
        self.args = args
        self.min_layer_limit = args.layer_min_limit
        self.max_layer_limit = args.layer_max_limit
        self.init_utility = args.init_utility 

    def enumerate_state(self, state, q_values):
        actions = []

        if state.terminate == 0:

            for latent_size in self.ssp.possible_latent_sizes:
                if state.layer_depth >= self.min_layer_limit:
                    actions += [State(layer_type=state.layer_type,
                                        layer_depth=state.layer_depth + 1,
                                        filter_depth=state.filter_depth,
                                        filter_size=state.filter_size,
                                        stride=state.stride,
                                        image_size=state.image_size,
                                        fc_size=latent_size,
                                        terminate=1)]
        
            if state.layer_depth < self.max_layer_limit:
                
                # Conv states -- iterate through all possible depths, filter sizes, and strides
                if (state.layer_type in ['start', 'conv', 'wrn']):  
                    for depth in self.ssp.possible_conv_depths:
                        for filt in self._possible_conv_sizes(state.image_size):
                            actions += [State(layer_type='conv',
                                                layer_depth=state.layer_depth + 1,
                                                filter_depth=depth,
                                                filter_size=filt,
                                                stride=2,
                                                image_size=state.image_size if self.ssp.conv_padding == 'SAME' \
                                                                            else self._calc_new_image_size(state.image_size, filt, 2),
                                                fc_size=0,
                                                terminate=0)]
                # WRNBasicBlock, same starting initial conditions as conv
                # TODO: stride fixed to 1, filter size fixed to 3
                if state.layer_type in ['conv', 'wrn']:
                    for depth in self.ssp.possible_conv_depths:
                        # TODO: hardcoded filter size of 3 again for possible wrn block
                        if state.image_size > 3:
                            actions += [State(layer_type='wrn',
                                              layer_depth=state.layer_depth + 2,
                                              filter_depth=depth,
                                              filter_size=3,
                                              stride=1,
                                              image_size=state.image_size,
                                              fc_size=0,
                                              terminate=0)]

        # Add states to transition and q_value dictionary
        q_values[state.as_tuple()] = {'actions': [to_state.as_tuple() for to_state in actions],
                                      'utilities': [self.init_utility for i in range(len(actions))]}
        return q_values        

    def transition_to_action(self, start_state, to_state):
        action = to_state.copy()
        return action

    def state_action_transition(self, start_state, action):
        ''' start_state: Should be the actual start_state, not a bucketed state
            action: valid action

            returns: next state, not bucketed
        '''
        to_state = action.copy()
        return to_state

    def bucket_state_tuple(self, state):
        bucketed_state = State(state_list=state).copy()
        bucketed_state.image_size = self.ssp.image_size_bucket(bucketed_state.image_size)
        return bucketed_state.as_tuple()

    def bucket_state(self, state):
        bucketed_state = state.copy()
        bucketed_state.image_size = self.ssp.image_size_bucket(bucketed_state.image_size)
        return bucketed_state

    def _calc_new_image_size(self, image_size, filter_size, stride = 2):
        '''Returns new image size given previous image size and filter parameters'''
        if ((image_size - filter_size)/float(stride) + 1)%2 != 0:
            new_size = int(math.floor(float(image_size - filter_size + 2) / float(stride) + 1))
            return new_size
        new_size = int(math.floor(float(image_size - filter_size) / float(stride) + 1))
        return new_size

    def _possible_conv_sizes(self, image_size):
        # TODO: using default argument 'stride' = 2 in self._calc_new_image_size()  
        return [conv for conv in self.ssp.possible_conv_sizes if (conv < image_size \
            and self._calc_new_image_size(image_size, conv) > 0)]

    def _possible_pool_sizes(self, image_size):
        return [pool for pool in self.ssp.possible_pool_sizes if pool < image_size]

    def _possible_pool_strides(self, filter_size):
        return [stride for stride in self.ssp.possible_pool_strides if stride <= filter_size]

    def _possible_fc_size(self, state):
        '''Return a list of possible FC sizes given the current state'''
        if state.layer_type=='fc':
            return [i for i in self.ssp.possible_fc_sizes if i <= state.fc_size]
        return self.ssp.possible_fc_sizes

    def allow_fully_connected(self, image_size, min_size):
        return image_size <= min_size

