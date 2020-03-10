import tensorflow as tf
import tensorflow.compat.v1 as tfc
#import tensorflow.compat.v1 as tfc
import numpy as np
from Environment_0 import VREP_env
import time
#import msvcrt
import sys
tfc.disable_eager_execution()

class Network(object):


    def __init__(self, env, scope, num_layers, num_units, obs_plc, act_plc, trainable=True):
        self.env = env
        self.action_size = 2
        self.trainable = trainable

        self.scope = scope

        self.obs_place = obs_plc
        self.acts_place = act_plc

        self.p, self.v, self.logstd = self._build_network(num_layers=num_layers, num_units=num_units)
        self.act_op = self.action_sample()

    def _build_network(self, num_layers, num_units):
        with tfc.variable_scope(self.scope):
            x = self.obs_place

            logstd = tfc.get_variable(name="logstd", shape=[self.action_size],
                                      initializer=tfc.zeros_initializer)

            for i in range(num_layers):
                x = tfc.layers.dense(x, units=num_units, activation=tf.nn.relu, name="px_fc" + str(i),
                                    trainable=self.trainable,use_bias=False)
            action = tfc.layers.dense(x, units=self.action_size, activation=tf.nn.tanh,
                                     name="pa1_fc" + str(num_layers), trainable=self.trainable,use_bias=False,)
            #x = tf.concat([self.obs_place,action],1)
            x = self.obs_place
            for i in range(num_layers):
                x = tfc.layers.dense(x, units=num_units, activation=tf.nn.relu, name="v_fc" + str(i),
                                    trainable=self.trainable,use_bias=False)
            value = tfc.layers.dense(x, units=1, activation=None, name="v_fc" + str(num_layers),
                                    trainable=self.trainable,use_bias=False)


        return action, value, logstd

    def action_sample(self):
        return self.p + 0.0*(tf.exp(self.logstd) * tfc.random_normal(tf.shape(self.p)))

    def get_variables(self):
        return tfc.get_collection(tfc.GraphKeys.GLOBAL_VARIABLES, self.scope)


class AutoRunAgent(object):


    def __init__(self, env):
        self.env = env

        # hyperparameters
        self.n_round = 15
        self.step_size = 10*self.n_round #<**>

        # placeholders
        self.adv_place = tfc.placeholder(shape=[None], dtype=tf.float32)
        self.return_place = tfc.placeholder(shape=[None], dtype=tf.float32)
        self.obs_place = tfc.placeholder(shape=[None, 20],
                                        name='ob', dtype=tf.float32)
        self.acts_place = tfc.placeholder(shape=[None, 2],
                                         name='ac', dtype=tf.float32)

        # build network
        self.net = Network(env=self.env,
                           scope="pi",
                           num_layers=4,
                           num_units=64,
                           obs_plc=self.obs_place,
                           act_plc=self.acts_place)

        self.saver = tfc.train.Saver()

    def traj_generator(self):

        t = 0

        env.setUp()
        
        action = [0,0]
        done = False
        ob = env.observe()


        cur_ep_return = 0
        cur_ep_length = 0
        ep_returns = []
        ep_lengths = []

        obs = np.array([ob for _ in range(self.step_size)])
        rewards = np.zeros(self.step_size, 'float32')
        values = np.zeros(self.step_size, 'float32')
        dones = np.zeros(self.step_size, 'int32')
        actions = np.array([action for _ in range(self.step_size)])
        prevactions = actions.copy()

        t_reset = 0
        round_reward = 0
        count_data = 0

        time_start = 0

        while True:
            prevaction = action
            while (time.time() - time_start < 0.05):
                pass

            action, value = self.act(ob)
            ob, reward, done = env.step(action[0])
            time_start = time.time()

            i = t % self.step_size

            obs[i] = ob
            values[i] = value
            dones[i] = done

            actions[i] = action[0]
            prevactions[i] = prevaction


            rewards[i] = reward
            cur_ep_return += reward

            cur_ep_length += 1
            
            if t > 0 and t_reset >= int((self.step_size/self.n_round)-1.0):
                count_data += int((self.step_size/self.n_round)-1.0)
                done = True


            if done:
                round_reward += cur_ep_return
                print("Reward: {}".format(cur_ep_return))
                ep_returns.append(cur_ep_return)
                ep_lengths.append(cur_ep_length)
                cur_ep_return = 0
                cur_ep_length = 0


                if t > 0 and count_data >= self.step_size:
                    print("Average Reward: {}".format(round_reward / self.n_round))
                    round_reward = 0.0
                    count_data = 0.0
                    yield {"ob": obs, "reward": rewards, "value": values,
                           "done": dones, "action": actions, "prevaction": prevactions,
                           "nextvalue": value * (1 - done), "ep_returns": ep_returns,
                           "ep_lengths": ep_lengths}

                t_reset = 0
                ep_returns = []
                ep_lengths = []
            t += 1
            t_reset += 1

    def act(self, ob):

        action, paction, value = tfc.get_default_session().run([self.net.act_op,self.net.p, self.net.v], feed_dict={
            self.net.obs_place: ob[None]
        })

        return action, value

    def run(self):
        traj_gen = self.traj_generator()

        while(True):
            traj = traj_gen.__next__()

    def restore_model(self, model_path):
        self.saver.restore(tfc.get_default_session(), model_path)
        print("model restored")


if __name__ == "__main__":

    env = VREP_env()
    sess = tfc.InteractiveSession()
    autoRun = AutoRunAgent(env)
    tfc.get_default_session().run(tfc.global_variables_initializer())
    autoRun.restore_model("./model/checkpointVREP_Real.ckpt") #--------
    autoRun.run()

    env.end()