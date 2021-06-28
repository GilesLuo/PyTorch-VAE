import logging
import random
import gym
import torch
import numpy as np
# from inception_score import inception_score
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
# todo: add image to rendering

class IS_Env(gym.Env):
    metadata = {
        'render.modes'           : ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self, state_dim, episode_len, dataset="CIFAR10", model_path="./model/vae.pth"):
        if dataset is "CIFAR10":
            self.dataset = dset.CIFAR10(root='data/', download=True,
                               transform=transforms.Compose([
                                   # transforms.Scale(32),
                                   transforms.ToTensor(),
                                   # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ]))
        else:
            raise NotImplementedError
        self.state_dim, self.episode_len, self.model_path = state_dim,  episode_len, model_path
        self.n_samples = len(self.dataset)
        self.actions = [0, 1]
        self.step_counter = None
        self.viewer = None
        self.state = None
        self.distribution, self.mean, self.logvar = None, None, None

    def _seed(self, seed=None):
        self.np_random, seed = random.seeding.np_random(seed)
        return [seed]

    def reset(self):

        random_index = int(np.random.random() * self.n_samples)
        self.selected_image = self.dataset[random_index][0]
        VAE_model = torch.load(self.model_path)
        self.step_counter = 0

        self.mean, self.logvar = VAE_model.encode(self.selected_image)
        cov = torch.exp(self.logvar)
        samples = np.random.multivariate_normal(self.mean, np.diag(cov), (self.episode_len,), 'raise')
        # copy[mean logvar] stack sample
        return samples

    def getNextState(self):

        return

    def step(self, action):


        return next_state, r, is_terminal, {}



    def render(self, mode='human'):
        pass
        # from gym.envs.classic_control import rendering
        # screen_width = 600
        # screen_height = 600
        #
        #     self.viewer.add_geom(self.fire1)
        #     self.viewer.add_geom(self.fire2)
        #     self.viewer.add_geom(self.diamond)
        #     self.viewer.add_geom(self.robot)
        #
        # if self.state is None:
        #     return None
        #
        # self.robotrans.set_translation(self.x[self.state-1], self.y[self.state- 1])
        #
        # return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()


if __name__== "__main__":
    env = IS_Env(state_dim=20, episode_len=10, dataset="CIFAR10")
    # print(env.dataset[0][0].shape)
    print(env.reset())