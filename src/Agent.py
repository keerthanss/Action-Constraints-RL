from Action import Action
from Gridworld import GridWorld

import numpy as np

class Agent:

    def __init__(self, state_dims, num_actions, lr=1e-2, epsilon=0, gamma=1, decay_lr = 0.99, decay_epsilon=0.75, supervisor=False):
        # state_dims is a list of dim sizes for each dim of
        # the state
        self.num_indices = len(state_dims)
        self.dim_sizes = state_dims
        # useful for index calculation
        self.cum_mult = np.array([np.prod(state_dims[i+1:]) for i in range(self.num_indices)])
        # print(self.dim_sizes, self.cum_mult)

        total_state_dim = np.prod(state_dims)
        # self.Qtable = np.random.rand(total_state_dim, num_actions)
        self.Qtable = np.zeros((total_state_dim, num_actions))

        self.num_actions = num_actions
        self.lr = lr
        if epsilon == 0:
            self.epsilon =  0.5 / (num_actions - 1) #50% chance of picking best action
        else:
            self.epsilon = epsilon
        self.gamma = gamma
        self.decay_lr = decay_lr
        self.decay_epsilon = decay_epsilon

        self.supervisor = supervisor
        self.neginf = -100000
        return

    def _state_index(self, s):
        return int(np.sum(np.array(s) * self.cum_mult))

    # helper function to access Q table values
    def Q(self, s, a):
        a_int = int(a)
        s_index = self._state_index(s)
        return self.Qtable[s_index, a_int]

    # helper function to obtain best action and its Q value
    def Qmax(self,s):
        s_index = self._state_index(s)
        # print(s, s_index)
        view = self.Qtable[s_index]
        action = np.argmax(view)
        return action, view[action]

    # greedy policy
    def Pi(self, s, env):
        prob_wts = [self.epsilon] * self.num_actions
        best_action, _ = self.Qmax(s)
        prob_wts[best_action] = 1 - (self.epsilon * (self.num_actions - 1))

        if self.supervisor:
            # Supervisor control
            s_index = self._state_index(s)
            legal, total_illegal_prob = [True]*self.num_actions, 0
            for action in range(self.num_actions):
                check_illegal = env.step(action, commit=False)
                if check_illegal:
                    # print(s,action, " is Illegal")
                    legal[action] = False
                    total_illegal_prob += prob_wts[action]
                    prob_wts[action] = 0
                    self.Qtable[s_index, int(action)] = self.neginf

            num_legal = np.sum(legal)
            if num_legal == 0:
                # if all actions are illegal, then let the env give a huge
                # negative reward and end the trajectory
                # Any arbitrary action can be given
                return 0

            # redistribute probability weights
            increment = total_illegal_prob / num_legal
            for action in range(self.num_actions):
                if legal[action]:
                    prob_wts[action] += increment

        return Action(np.random.choice(self.num_actions, p=prob_wts))

    # Q learning algorithm update rule
    def update(self, s, a, reward, sprime, done):
        # done - bool
        done = int(done)
        s_index = self._state_index(s)
        sprime_index = self._state_index(sprime)
        a_int = int(a)
        delta = reward + (1-done) * (self.gamma * self.Qmax(sprime)[1] - self.Q(s,a))
        self.Qtable[s_index, a_int] += self.lr * delta
        return

    def decay(self):
        self.epsilon = self.decay_epsilon * self.epsilon
        self.lr = self.decay_lr * self.lr
        return
