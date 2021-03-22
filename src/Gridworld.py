from Action import Action
from NonMarkovian import ActionConstraint, ActionRectification, RewardMachine

from functools import reduce
from enum import IntEnum

import numpy as np

class Wall:
    '''
    A simple container class that describes walls in the
    gridworld.
    '''
    def __init__(self, *args):
        if len(args) == 1:
            self._merge_constructor(args[0])
        elif len(args) == 3:
            self._default_constructor(args[0], args[1], args[2])
        else:
            raise NotImplementedError
        return


    def _default_constructor(self, h_or_w, axis, values):
        '''
        h_or_w provides a string by which we can interpret
        axis to either be across the rows or across the columns.
        For example: Suppose we want to specify a wall that's
        along the 3rd column line in the gridworld, and spanning
        across the top 2 cells.
        We would do so as
        Wall('h', 2, [5,6])
        Wall object will then internally store all transitions
        across this wall that will be forbidden.
        '''
        assert (h_or_w == 'h' or h_or_w == 'w'), \
                    "Invalid character passed to Wall"
        length= len(values)
        if h_or_w == 'h':
            # By default it is assumed that the wall is to the
            # RIGHT of the axis
            points_to_left_of_wall = list(zip(values, [axis]*length))
            points_to_right_of_wall = list(zip(values, [axis+1]*length))
            wall = list(zip(points_to_left_of_wall, points_to_right_of_wall))
            # store mirror of the same for convenient checking later
            wall = wall + list(zip(points_to_right_of_wall, points_to_left_of_wall))
        elif h_or_w == 'w':
            # By default it is assumed that the wall is ABOVE the
            # axis
            points_below_wall = list(zip([axis]*length, values))
            points_above_wall = list(zip([axis+1]*length, values))
            wall = list(zip(points_below_wall, points_above_wall))
            # store mirror of same for convenient checking later
            wall = wall + list(zip(points_above_wall, points_below_wall))
        self.wall = set(wall)
        return

    def _merge_constructor(self, walls_list):
        walls_sets = map(lambda x : x.wall, walls_list)
        full_union = reduce(lambda x,y : x.union(y), walls_sets)
        self.wall = full_union
        return

    def collision(self, pos1, pos2):
        # we only need to check if this transition has already
        # been listed by self.wall
        return ((pos1, pos2) in self.wall)

class Orientation(IntEnum):
    North = 0
    South = 1
    East = 2
    West = 3

class GridWorld:
    '''
    Gridworld described by coordinates as so

    | (2,0) | (2,2) | (2,2) |
    | (1,0) | (1,1) | (1,2) |
    | (0,0) | (0,1) | (0,2) |

    '''
    def __init__(self, randomize=False, check_walls=True, keep_interrupt=True):
        self.h = 7
        self.w = 9
        # self.h = 4
        # self.w = 5
        wall_constraints = [Wall('h',2,[0,1,5,6]), \
                                    Wall('h',3,[0,1,5,6]), \
                                    Wall('h', 5, [3]),\
                                    Wall('h', 6, [3]),\
                                    Wall('w', 2, [0,1,5,7]),\
                                    Wall('w', 3, [0,1,5,7])]
        self.wall = Wall(wall_constraints) # will merge them into one

        if not randomize:
            self.fruits = [(6, 0), (6, 8), (3, 6)]
            self.start_pos = (0,6)
            # self.fruits = [(3,0), (3,4), (0,4)]
            # self.start_pos = (0,0)
        else:
            raise NotImplementedError
        self.ac = ActionConstraint()
        self.rm = RewardMachine()
        self.ar = ActionRectification()
        # self.ar = ActionRectification()
        self.current_pos = self.start_pos

        self.fruit_picked = [0,0,0]
        self.orientation = Orientation.North

        # mapping from (orientation, Action) to (position increment, new orientation)
        self.position_change = {
        (Orientation.North, Action.Right) : ((0,1), Orientation.East), (Orientation.North, Action.Left) : ((0,-1),Orientation.West),
        (Orientation.North, Action.Forward): ((1,0), Orientation.North), (Orientation.North, Action.Backward) : ((-1,0), Orientation.North),
        (Orientation.South, Action.Right) : ((0,-1), Orientation.West), (Orientation.South, Action.Left) : ((0,1),Orientation.East),
        (Orientation.South, Action.Forward): ((-1,0), Orientation.South), (Orientation.South, Action.Backward) : ((1,0), Orientation.South),
        (Orientation.East, Action.Left) : ((1,0),Orientation.North), (Orientation.East, Action.Right) : ((-1,0),Orientation.South),
        (Orientation.East, Action.Forward): ((0,1), Orientation.East), (Orientation.East, Action.Backward) : ((0,-1), Orientation.East),
        (Orientation.West, Action.Left): ((-1,0),Orientation.South), (Orientation.West, Action.Right) : ((1,0), Orientation.North),
        (Orientation.West, Action.Forward): ((0,-1), Orientation.West), (Orientation.West, Action.Backward) : ((0,1), Orientation.West)
        }

        self.state = self.make_state()
        self.done = False

        self.action_size = 4

        self.check_walls = check_walls
        self.keep_interrupt = keep_interrupt

        self.interrupt = False # an indicator that activates after all fruits have been picked
        self.drop_probability = 0.5 # drop probability after all 3 fruits have been picked
        self.times_dropped = 0
        return

    def reset(self):
        self.current_pos = self.start_pos
        self.ac.reset()
        self.rm.reset()
        self.ar.reset()
        self.fruit_picked = [0,0,0]
        self.done = False
        self.orientation = Orientation.North
        self.interrupt = False
        self.times_dropped = 0
        self.state = self.make_state()
        return self.state

    def get_state_dims(self):
        state_dims = [self.h, self.w, 4, 4, 6]
        return state_dims

    def make_state(self):
        # position + indicator of whether fruits have been picked +
        # state variables of all non markovian constraints
        return list(self.current_pos) + [int(self.orientation), self.ac.get_state(), self.rm.get_state()]

    def _transition(self, pos, action):
        # increment = (\
        #         0 + (action==Action.Forward) - (action==Action.Backward),\
        #         0 + (action==Action.Right) - (action==Action.Left)\
        #         )
        increment, new_orientation = self.position_change[(self.orientation, action)]
        new_pos = (increment[0]+pos[0], increment[1]+pos[1])
        return new_pos, new_orientation

    def _bounded(self, pos):
        return (pos[0] < self.h and pos[1] < self.w and pos[0] >=0 and pos[1] >= 0)

    def _is_illegal_state_transition(self, s1,s2):
        bounds_overstep = not self._bounded(s2)
        wall_collision = False
        if self.check_walls:
            wall_collision = self.wall.collision(s1,s2)
        return bounds_overstep or wall_collision

    def is_task_completed(self):
        return (self.current_pos == self.start_pos and self.fruit_picked == [1,1,1])

    def _is_terminal(self):
        if self.is_task_completed():
            return True
        if self.ac.marked() or self.rm.marked():
            return True
        return False

    def interrupt_service_routine(self):
        action_dict = {0: None, 1:Action.Drop2, 2:Action.Drop3, 3:Action.Pick1, 4:Action.Pick2, 5:Action.Pick3}
        fruit_dict = {1:[0,0,1], 2:[0,0,0], 3:[1,0,0], 4:[1,1,0], 5:[1,1,1]}
        overridden = False
        if self.fruit_picked == [1,1,1] or self.ar.is_active():
            if self.interrupt == False:
                # wait one step before possibly dropping
                self.interrupt = True
            else:
                # check state of ActionRectification
                if not self.ar.is_active():
                    probs = [1-self.drop_probability, self.drop_probability]
                    to_drop_or_not_to_drop = np.random.choice(2,p=probs)
                    if to_drop_or_not_to_drop == 1:
                        # drop
                        state = self.ar.get_state()
                        action = Action.Drop1
                        self.ar.activate()
                        self.fruit_picked = [0, 1, 1]
                        self.times_dropped += 1
                        overridden = True
                else:
                    # it has already been dropped
                    state = self.ar.get_state()
                    action = action_dict[state]
                    self.ar.transition(action)
                    self.fruit_picked = fruit_dict[state]
                    if not self.ar.is_active():
                        # interrupt service has completed
                        self.interrupt = False
                    overridden = True
        return overridden

    def step(self, action, commit=True):
        if self.done:
            return self.state, 0, self.done, False
        # transition state
        new_pos, new_orientation = self._transition(self.current_pos, action)
        # check if transition is allowed
        illegal_transition = self._is_illegal_state_transition(self.current_pos, new_pos)

        if not commit:
            # query FSA w.r.t action constraints
            outcome = self.ac.transition(action, commit)
            # return whether transition is legal
            return (illegal_transition or outcome == 3) # Bad code. TODO

        if self.keep_interrupt:
            override = self.interrupt_service_routine()
            if override:
                return self.state, self.rm.reward(), self.done, override
        else:
            override = False

        # commit changes
        if illegal_transition:
            new_pos = self.current_pos
            new_orientation = self.orientation

        # transition FSA w.r.t. action constraints
        self.ac.transition(action)
        self.current_pos = new_pos
        self.orientation = new_orientation
        # pick up fruits if at location and unpicked
        for i in range(3):
            if self.fruit_picked[i] == 1:
                continue
            elif self.current_pos == self.fruits[i]:
                self.fruit_picked[i] = 1

        # transition Reward Machine
        self.rm.transition(self.fruit_picked, self.current_pos == self.start_pos)

        # check terminal
        self.done = self._is_terminal()
        # make new state
        self.state = self.make_state()
        # calculate reward
        reward = 0
        if illegal_transition:
            reward += -1000
        if self.ac.marked():
            reward += -1000
        if self.rm.marked():
            reward += -1000
        reward += self.rm.reward()
        return self.state, reward, self.done, override
