from toll_plaza_env import TollPlazaEnv

env = TollPlazaEnv()
obs, info = env.reset()
print(env.step(env.action_space.sample()))
