"""
Double DQN (Nature 2015)
http://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf
Notes:
    The difference is that now there are two DQNs (DQN & Target DQN)
    y_i = r_i + 𝛾 * max(Q(next_state, action; 𝜃_target))
    Loss: (y_i - Q(state, action; 𝜃))^2
    Every C step, 𝜃_target <- 𝜃
"""
import numpy as np
import pandas as pd 
import tensorflow as tf
import random
from collections import deque
import dqn

from typing import List

# Constants defining our neural network
DISCOUNT_RATE = 0.8
REPLAY_MEMORY = 100
BATCH_SIZE = 32
TARGET_UPDATE_FREQUENCY = 5
MAX_EPISODES = 1000


def replay_train(mainDQN: dqn.DQN, targetDQN: dqn.DQN, train_batch: list):
    """Trains `mainDQN` with target Q values given by `targetDQN`
    Args:
        mainDQN (dqn.DQN): Main DQN that will be trained
        targetDQN (dqn.DQN): Target DQN that will predict Q_target
        train_batch (list): Minibatch of replay memory
            Each element is (s, a, r, s', done)
            [(state, action, reward, next_state, done), ...]
    Returns:
        float: After updating `mainDQN`, it returns a `loss`
    """
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])

    X = states

    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * ~done

    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target

    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y)


def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str):
    """Creates TF operations that copy weights from `src_scope` to `dest_scope`
    Args:
        dest_scope_name (str): Destination weights (copy to)
        src_scope_name (str): Source weight (copy from)
    Returns:
        List[tf.Operation]: Update operations are created and returned
    """
    # Copy variables src_scope to dest_scope
    op_holder = []

    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)

    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


if __name__ == "__main__":
    # Read Variables                                                                                                                        
    fn = '/Users/skyeong/data/adhd200/Results/adhd200.csv'
    T = pd.read_csv(fn,sep=',')
    states = np.array([T['Dx'],T['Gender'],T['D'],T['kappa'],T['Th'],T['Lf']])

    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    last_100_game_reward = deque(maxlen=100)


    # Constants defining our neural network
    INPUT_SIZE = 5
    OUTPUT_SIZE = 2

    with tf.Session() as sess:
        mainDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = dqn.DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(MAX_EPISODES):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0

            idx1 = np.random.permutation(states.shape[1])
            states1 = states[:,idx1]

            while not done:
                idx2 = np.random.permutation(states1.shape[1])
                Dx = states1[0,idx2[1]]
                state = states1[1:,idx2[0]]

                if np.random.rand() < e:
                    action = 1 if np.random.rand(1)>0.5 else 0
                else:
                    # Choose an action by greedily from the Q-network
                    action = np.argmax(mainDQN.predict(state))

                # Get new state and reward from environment
                reward = -1 if np.abs(action-Dx)>0.3 else 0
                done = True if np.abs(action-Dx)<0.1 else False
                next_state = states1[1:,idx2[1]]
                #print (state, action, done, reward)
                if done:  # Penalty
                    reward = 1

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                if step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops)

                state = next_state
                step_count += 1

            print("Episode: {}  steps: {}".format(episode, step_count))

            # CartPole-v0 Game Clear Checking Logic
            last_100_game_reward.append(step_count)

            if len(last_100_game_reward) == last_100_game_reward.maxlen:
                avg_reward = np.mean(last_100_game_reward)

                if avg_reward > 199:
                    print(f"Game Cleared in {episode} episodes with avg reward {avg_reward}")
                    break

    action = np.argmax(mainDQN.predict(state))