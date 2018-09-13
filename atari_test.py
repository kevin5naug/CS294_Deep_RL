import gym
import time
env=gym.make('SpaceInvaders-v0')
env.reset()
env.render()
time.sleep(5)
env.env.close()
