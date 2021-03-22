from Gridworld import GridWorld
from Agent import Agent
from Action import Action

import sys
import numpy as np
import pickle
import argparse

class Logger:

    def __init__(self, freq):
        self.best_score = -10000
        self.prev_score = np.zeros(freq, dtype=np.int16)
        self.prev_steps = np.zeros(freq, dtype=np.int16)
        self.prev_fruits = np.zeros((3, freq), dtype=np.bool)
        self.fruit_drop = np.zeros(freq, dtype=np.int8)
        self.task_completion = np.zeros(freq, dtype=np.bool)
        print("Epoch, Average Steps, Average Score, Fruit 1%, Fruit 2%, Fruit 3%, Avg Fruit Drop, Num Task Completion, Best score yet")
        self.freq = freq

    def update(self, epoch, score, steps, env):
        idx = epoch % self.freq
        self.best_score = max(self.best_score, score)
        # idx = epochs % print_freq
        self.prev_score[idx] = score
        self.prev_steps[idx] = steps
        self.prev_fruits[0][idx] = env.fruit_picked[0]
        self.prev_fruits[1][idx] = env.fruit_picked[1]
        self.prev_fruits[2][idx] = env.fruit_picked[2]
        self.task_completion[idx] = env.is_task_completed()
        self.fruit_drop[idx] = env.times_dropped

    def log(self, epoch):
        average_score = np.sum(self.prev_score) / self.freq
        fruits_picked = np.sum(self.prev_fruits, axis=1) * 100 / self.freq
        average_steps = np.sum(self.prev_steps) / self.freq
        count_task_completion = np.sum(self.task_completion)
        average_fruit_drop = np.sum(self.fruit_drop) / self.freq
        print("{}, {}, {}, {}, {}, {}, {}, {}, {}".format(epoch+1, average_steps, average_score, fruits_picked[0], \
                                    fruits_picked[1], fruits_picked[2], average_fruit_drop, count_task_completion, self.best_score))
        return


def generate_best_trajectory(env, agent):
    s, done, traj, steps = env.reset(), False, [], 0
    while not done and steps < 60:
        a, _ = agent.Qmax(s)
        sprime, r, done, interrupt = env.step(a)
        if interrupt:
            action_dict = {0: Action.Drop1, 1:Action.Drop2, 2:Action.Drop3, 3:Action.Pick1, 4:Action.Pick2, 5:Action.Pick3}
            s = env.ar.get_state() - 1 if env.ar.is_active() else 5
            a = action_dict[env.ar.get_state()]
        traj.append([s,a,r, done])
        s = sprime
        steps += 1
    return traj


def main(lr, epsilon, gamma, decay_lr, decay_epsilon, modelfile):
    seed = 42
    np.random.seed(seed)
    env = GridWorld(check_walls=True)
    # env = GridWorld(check_walls=False)
    agent = Agent(env.get_state_dims(), env.action_size, lr, epsilon, gamma, decay_lr, decay_epsilon, supervisor=True)
    exp_name = "gridworld_lr{}_ep{}_gamma{}_decaylr{}_decayep{}_s{}".format(lr, epsilon, gamma, decay_lr, decay_epsilon, seed)

    print_freq = 10000
    logger = Logger(print_freq)

    for epochs in range(500000):
        s, done, trajectory, score, steps = env.reset(), False, [], 0, 0
        print_traj = False
        while not done and steps < 60: #100:
            a = agent.Pi(s, env)
            sprime, r, done, interrupt = env.step(a)
            a2 = a
            if not interrupt:
                agent.update(s, a, r, sprime, done)
            else:
                action_dict = {0: Action.Drop1, 1:Action.Drop2, 2:Action.Drop3, 3:Action.Pick1, 4:Action.Pick2, 5:Action.Pick3}
                s = env.ar.get_state() - 1 if env.ar.is_active() else 5
                a2 = action_dict[s]
                # print_traj = True

            trajectory.append([s,a,a2, r, sprime, done])
            s = sprime
            score += r
            steps += 1

        if print_traj:
            print(trajectory)
        logger.update(epochs, score, steps, env)
        if epochs % print_freq == print_freq - 1:
            logger.log(epochs)
            print(generate_best_trajectory(env, agent))
            agent.decay()

    f = open(modelfile, "wb")
    pickle.dump(agent,f)
    f.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser("Gridworld with Non-Markovian Task and Action Constraints")
    parser.add_argument("--lr", type=float, help="Learning Rate", default=0.01)
    parser.add_argument("--epsilon", type=float, help="The epsilon probability assigned to picking actions other than the best one. \
                                        The default value calibrates such that best action receives 50% probability.", default=0)
    parser.add_argument("--gamma", type=float, help="Discount rate", default=1)
    parser.add_argument("--decay_lr", type=float, help="Decay rate for learning rate", default=1)
    parser.add_argument("--decay_epsilon", type=float, help="Decay rate for epsilon", default=1)
    parser.add_argument("--model_file", type=str, help="Name of the pickle file with which to save the Agent", default="model.pickle")
    args = parser.parse_args()
    main(args.lr, args.epsilon, args.gamma, args.decay_lr, args.decay_epsilon, args.model_file)
