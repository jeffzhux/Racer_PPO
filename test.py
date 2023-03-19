
from gym_unity.envs import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment

env_name = './Racer/Racer.exe'
unity_env = UnityEnvironment(file_name=env_name)
env = UnityToGymWrapper(unity_env=unity_env, allow_multiple_obs=False)

for i in range(2):
    obs = env.reset()
    done = False
    while not done:

        # 執行動作後的 狀態, 獎勵, 是否結束

        next_obs, rew, done, info = env.step([1,0])
        obs = next_obs
        print(done)
    print(obs)
env.close()