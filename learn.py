from agent import SacAgent
import gym
import pybullet_envs
import argparse
import csv


def learn(device=0, environment=0, log=1):
    env = gym.make(env_list[environment])
    log_dir = 'saves/' + str(environment+1) + '/log' + str(log) + '.csv'
    with open(log_dir, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['frames', 'return'])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SacAgent(state_dim, action_dim, device=device)
    total_frames = 0
    while total_frames < 2.01e6:
        state = env.reset()
        frame = 0
        while 1:
            if total_frames < 1e4:
                action = env.action_space.sample()
            else:
                action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            frame += 1
            total_frames += 1
            agent.remember(state, next_state, action, reward, done and frame < 1e3)
            if total_frames > 1e4:
                agent.train()
            state = next_state
            if total_frames >= 1e4 and total_frames % 1e3 == 0:
                with open(log_dir, "a+", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([total_frames - 1e4, test(environment, agent)])
            if done:
                break


def test(environment, agent):
    env = gym.make(env_list[environment])
    state = env.reset()
    total_reward = 0
    while 1:
        action = agent.act(state, mean=True)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        if done:
            break
    return total_reward


if __name__ == '__main__':
    env_list = ['Hopper-v2', 'Walker2d-v2', 'HalfCheetah-v2', 'Ant-v2', 'Humanoid-v2',
                'HopperBulletEnv-v0', 'Walker2DBulletEnv-v0', 'HalfCheetahBulletEnv-v0',
                'AntBulletEnv-v0', 'HumanoidBulletEnv-v0']
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-e', '--env', type=int, default=0)
    parser.add_argument('-l', '--log', type=int, default=1)
    args = parser.parse_args()
    learn(device=args.gpu, environment=args.env, log=args.log)
