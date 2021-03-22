from Action import Action

from functools import reduce

class ActionConstraint:

    def __init__(self):
        # states are 0,1,2,3
        # state 3 is marked
        self.state = 0

    def transition(self,action, commit=True):
        prev_state = self.state

        if (action in [Action.Forward, Action.Backward]):
            if (self.state != 3):
                self.state = 0 #move back to 0
            else:
                self.state = 3 #stay in 3
        elif action == Action.Left:
            if self.state == 0:
                self.state = 2
            elif self.state == 1:
                self.state = 2
            else: # whether in 2 or 3, move to 3
                self.state = 3
        elif action == Action.Right:
            if self.state == 0:
                self.state = 1
            elif self.state == 2:
                self.state = 1
            else: # whether in 1 or 3, move to 3
                self.state = 3

        if not commit:
            # revert changes
            outcome = self.state
            self.state = prev_state
            return outcome

        return self.state

    def get_state(self):
        return self.state

    def marked(self):
        return self.state == 3

    def reset(self):
        self.state = 0

class ActionRectification:

    def __init__(self):
        # states 0 - 6
        # state 0 corresponds to an inactive state
        self.state = 0
        return

    def activate(self):
        if self.state == 0:
            self.state = 1
        return

    def get_state(self):
        return self.state

    def transition(self, action, commit=True):
        prev_state = self.state
        if self.state == 0:
            self.state = 0
        else:
            if self.state == 1 and action == Action.Drop2:
                self.state = 2
            elif self.state == 2 and action == Action.Drop3:
                self.state = 3
            elif self.state == 3 and action == Action.Pick1:
                self.state = 4
            elif self.state == 4 and action == Action.Pick2:
                self.state = 5
            elif self.state == 5 and action == Action.Pick3:
                self.state = 0 # complete, deactivate
            else:
                self.state = 6 # fail

        if not commit:
            outcome = self.state
            self.state = prev_state
            return outcome
        return self.state

    def marked(self):
        return self.state == 6

    def reset(self):
        self.state = 0

    def is_active(self):
        return self.state != 0


class RewardMachine:

    def __init__(self):
        # states are 0-5
        # each state has an associated reward
        self.state = 0
        self.reward_dict = {0:-10, 1:-20, 2:-8, 3:-6, 4:-1, 5:0}

    def reward(self):
        return self.reward_dict[self.state]

    def get_state(self):
        return self.state

    def transition(self, fruit_picked, at_terminal):
        label = reduce(lambda x,y: str(x)+str(y), fruit_picked)

        if label=="000":
            self.state = 0
        elif label=="100":
            self.state = 2
        elif label=="110":
            self.state = 3
        elif label=="111":
            if at_terminal:
                self.state = 5
            else:
                self.state = 4
        else:
            self.state = 1
        return

    def marked(self):
        return (self.state == 1)

    def reset(self):
        self.state = 0
