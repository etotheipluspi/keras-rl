import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import MAMemory


from madrl_environments.pursuit import PursuitEvade
from madrl_environments.pursuit.utils import TwoDMaps


ENV_NAME = 'Pursuit'

n_evaders = 10
n_pursuers = 6
obs_range = 5
map_pool = np.load("/Users/megorov/Desktop/projects/stanford/deeprl/madrl_problems/runners/maps/map_pool16.npy")
flatten = True
reward_mech = 'local'

rect_map = TwoDMaps.rectangle_map(10, 10)
map_pool = [rect_map]

env = PursuitEvade(map_pool, n_evaders=n_evaders, n_pursuers=n_pursuers,
                       obs_range=obs_range,
                       surround=True, 
                       sample_maps=True,
                       flatten=flatten,
                       reward_mech=reward_mech,
                       catchr=0.01,
                       term_pursuit=1.0)

nb_actions = env.action_space.n
# Model
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(132))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())


memory = MAMemory(limit=50000)
policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)

dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit_ma(env, nb_steps=1000000, visualize=False, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)
