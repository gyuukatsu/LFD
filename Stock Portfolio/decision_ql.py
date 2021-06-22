from tensorflow.python.platform import gfile
import tensorflow as tf
import numpy as np
import random

def softmax(action_q_vals):
    rate_a=action_q_vals / np.sum(action_q_vals)
    exp_a=np.exp(rate_a)
    soft_sum=np.sum(exp_a)
    soft=exp_a/soft_sum
    return soft


class QLearningDecisionPolicy:
    def __init__(self, epsilon, gamma, lr, actions, input_dim, model_dir):
        # select action function hyperparameter
        self.epsilon = epsilon
        # q functions hyperparameter
        self.gamma = gamma
        # neural network hyperparmeter
        self.lr = lr

        self.actions = actions
        output_dim = len(actions)

        # neural network input and output placeholder
        self.x = tf.compat.v1.placeholder(tf.float32, [None, input_dim])
        self.y = tf.compat.v1.placeholder(tf.float32, [output_dim])

        # TODO: build your Q-network
        # 2-layer fully connected network
        self.fc = tf.layers.dense(self.x, 15, activation=tf.nn.relu)
        self.fc2 = tf.layers.dense(self.fc, 15, activation=tf.nn.relu)
        self.q = tf.layers.dense(self.fc2, output_dim)

        # loss
        loss = tf.square(self.y - self.q)

        # train operation
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss)

        # session
        self.sess = tf.compat.v1.Session()

        # initalize variables
        init_op = tf.compat.v1.global_variables_initializer()
        self.sess.run(init_op)

        # saver
        self.saver = tf.compat.v1.train.Saver()

        # restore model
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            print("load model: %s" % ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def select_action(self, current_state, is_training=True):
        action_q_vals = self.sess.run(self.q, feed_dict={self.x: current_state})
        action_select=[]
        action_q_vals_select=[]
        cnt=0
        if random.random() >= self.epsilon or not is_training:
            for i in range(len(action_q_vals[0,:])-1):
                if action_q_vals[0,i]>0:
                    action_select.append(self.actions[i])
                    action_q_vals_select.append(action_q_vals[0,i])
                else:
                    cnt=cnt+1
            if cnt==(len(action_q_vals[0,:])-1):
                action_select.append(self.actions[-1])
                action_q_vals_select.append(action_q_vals[0][-1])
        else:  # randomly select action
            action_select.append(self.actions[random.randint(0, len(self.actions)-1)])
            action_q_vals_select.append(action_q_vals[0][self.actions.index(action_select[0])])
        return action_select, action_q_vals_select

    def update_q(self, current_state, action, reward, next_state, action_q_vals, port_rate, current_portfolio, criteria):
        # Q(s', a')
        next_action_q_vals = self.sess.run(self.q, feed_dict={self.x: next_state})
        next_action_list=[]
        next_action_q_vals_list=[]
        e_next_action_q_vals=0
        # a' index
        for i in range(len(next_action_q_vals[0,:])):
            if next_action_q_vals[0,i]>=0:
                next_action_list.append(self.actions[i])
                next_action_q_vals_list.append(next_action_q_vals[0,i])
            else:
                pass
        next_port_rate=softmax(next_action_q_vals_list)
        for i in range(len(next_port_rate)):
            e_next_action_q_vals+=next_port_rate[i]*next_action_q_vals_list[i]
        
        # create target
        action_q_vals[0, self.actions.index(action)] = 0.5*action_q_vals[0, self.actions.index(action)]+0.5*((reward-criteria*port_rate)/(current_portfolio*port_rate)*100 +self.gamma*(e_next_action_q_vals-sum(next_action_q_vals[0,:])/len(next_action_q_vals[0,:])))
        '''
        for i in range(len(action_q_vals[0,:])):
            if i==self.actions.index(action):
                pass
            else:
                action_q_vals[0,i]=action_q_vals[0,i]+(c-action_q_vals[0, self.actions.index(action)])/11
        '''
        # delete minibatch dimension
        
        
        return action_q_vals
    def save_model(self, output_dir):
        if not gfile.Exists(output_dir):
            gfile.MakeDirs(output_dir)

        checkpoint_path = output_dir + '/model'
        self.saver.save(self.sess, checkpoint_path)
    
    def restore_model(self, input_dir):
        checkpoint_path=input_dir + '/model.ckpt'
        self.restore_saver.restore(self.sess, checkpoint.path)