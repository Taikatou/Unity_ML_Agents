# 라이브러리 불러오기
import numpy as np
import datetime
import tensorflow as tf
import tensorflow.contrib as tc

from mlagents.envs import UnityEnvironment

from MADDPG.util.replay_buffer import ReplayBuffer


class MADDPG:
    def __init__(self, name, layer_norm=True, nb_actions=3, nb_input=36, nb_other_aciton=4):
        def actor_network(actor_name):
            with tf.variable_scope(actor_name) as scope:
                x = state_input
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, self.nb_actions,
                                    kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
                x = tf.nn.tanh(x)
            return x

        def critic_network(critic_name, action_input, reuse=False):
            with tf.variable_scope(critic_name) as scope:
                if reuse:
                    scope.reuse_variables()

                x = state_input
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.concat([x, action_input], axis=-1)
                x = tf.layers.dense(x, 64)
                if self.layer_norm:
                    x = tc.layers.layer_norm(x, center=True, scale=True)
                x = tf.nn.relu(x)

                x = tf.layers.dense(x, 1, kernel_initializer=tf.random_uniform_initializer(minval=-3e-3, maxval=3e-3))
            return x

        self.layer_norm = layer_norm
        self.nb_actions = nb_actions
        state_input = tf.placeholder(shape=[None, nb_input], dtype=tf.float32)
        action_input = tf.placeholder(shape=[None, nb_actions], dtype=tf.float32)
        other_action_input = tf.placeholder(shape=[None, nb_other_aciton], dtype=tf.float32)
        reward = tf.placeholder(shape=[None, 1], dtype=tf.float32)

        self.action_output = actor_network(name + '_actor')

        print("#####################################################################################################")
        print("##########                                   action                                     #############")
        print("#####################################################################################################")
        print(self.action_output)
        self.critic_output = critic_network(name + '_critic',
                                            action_input=tf.concat([action_input, other_action_input], axis=1))
        self.state_input = state_input
        self.action_input = action_input
        self.other_action_input = other_action_input
        self.reward = reward

        self.actor_optimizer = tf.train.AdamOptimizer(1e-4)
        self.critic_optimizer = tf.train.AdamOptimizer(1e-3)

        self.actor_loss = -tf.reduce_mean(
            critic_network(name + '_critic', action_input=tf.concat([self.action_output, other_action_input], axis=1),
                           reuse=True))
        self.actor_train = self.actor_optimizer.minimize(self.actor_loss)

        # avs = self.actor_optimizer.compute_gradients(self.actor_loss)
        # aapped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in avs if grad is not None]
        # self.actor_train = self.actor_optimizer.apply_gradients(aapped_gvs)

        self.target_Q = tf.placeholder(shape=[None, 1], dtype=tf.float32)
        self.critic_loss = tf.reduce_mean(tf.square(self.target_Q - self.critic_output))
        self.critic_train = self.critic_optimizer.minimize(self.critic_loss)

        # cvs = self.critic_optimizer.compute_gradients(self.critic_loss)
        # capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in cvs if grad is not None]
        # self.critic_train = self.actor_optimizer.apply_gradients(capped_gvs)

    def train_actor(self, state, other_action, sess):
        sess.run(self.actor_train, {self.state_input: state, self.other_action_input: other_action})

    def train_critic(self, state, action, other_action, target, sess):
        sess.run(self.critic_train,
                 {self.state_input: state, self.action_input: action, self.other_action_input: other_action,
                  self.target_Q: target})

    def action(self, state, sess):
        print("#####################################################################################################")
        print("##########                                   STATE                                      #############")
        print("#####################################################################################################")
        print(state)
        print("#####################################################################################################")
        print("##########                                ACTION OUTPUT                                 #############")
        print("#####################################################################################################")
        print(self.action_output)
        print("#####################################################################################################")
        print("##########                               STATE INPUT                                    #############")
        print("#####################################################################################################")
        print(self.state_input)
        return sess.run(self.action_output, {self.state_input: state})

    def Q(self, state, action, other_action, sess):
        return sess.run(self.critic_output,
                        {self.state_input: state, self.action_input: action, self.other_action_input: other_action})


class MADDPGLearner:
    def __init__(self, sess, name):
        def create_init_update(one_line_name, target_name, tau=0.99):
            online_var = [i for i in tf.trainable_variables() if one_line_name in i.name]
            target_var = [i for i in tf.trainable_variables() if target_name in i.name]

            target_init = [tf.assign(target, online) for online, target in zip(online_var, target_var)]
            target_update = [tf.assign(target, (1 - tau) * online + tau * target) for online, target in
                             zip(online_var, target_var)]

            return target_init, target_update

        self.agent = MADDPG(name + '_agent')
        self.target = MADDPG(name + '_critic')

        self.agent_actor_target_init, self.agent_actor_target_update = create_init_update('agent1_actor',
                                                                                          'agent1_target_actor')
        self.agent_critic_target_init, self.agent_critic_target_update = create_init_update('agent1_critic',
                                                                                            'agent1_target_critic')
        sess.run([self.agent_actor_target_init, self.agent_critic_target_init])

        self.memory = ReplayBuffer(100000)

    def get_agent_action(self, o_n, sess, noise_rate=0):
        print(o_n)
        action = self.agent.action(state=o_n, sess=sess) + np.random.randn(2) * noise_rate
        return action


def train_agent(learner, sess, other_actors):
    total_obs_batch, total_act_batch, rew_batch, total_next_obs_batch, done_mask = learner.memory.sample(32)

    act_batch = total_act_batch[:, 0, :]
    other_act_batch = np.hstack([total_act_batch[:, 1, :], total_act_batch[:, 2, :]])

    obs_batch = total_obs_batch[:, 0, :]

    next_obs_batch = total_next_obs_batch[:, 0, :]
    next_other_actor1_o = total_next_obs_batch[:, 1, :]

    next_other_action = np.hstack([other_actors[0].action(next_other_actor1_o, sess)])
    target = rew_batch.reshape(-1, 1) + 0.9999 * learner.target.Q(state=next_obs_batch,
                                                                  action=learner.agent.action(next_obs_batch, sess),
                                                                  other_action=next_other_action, sess=sess)
    learner.agent.age.train_actor(state=obs_batch, other_action=other_act_batch, sess=sess)

    learner.agent.train_critic(state=obs_batch,
                               action=act_batch,
                               other_action=other_act_batch,
                               target=target,
                               sess=sess)

    sess.run([learner.agent_actor_target_update, learner.agent_critic_target_update])


class EnvironmentTest():
    def __init__(self):
        # 유니티 환경 경로
        game = "Pong"
        env_name = "../../env/" + game + "/Windows/" + game

        env = UnityEnvironment(file_name=env_name)
        self.env = env

        # 유니티 브레인 설정
        self.brain_name1 = env.brain_names[0]
        self.brain_name2 = env.brain_names[1]

        self.brain1 = env.brains[self.brain_name1]
        self.brain2 = env.brains[self.brain_name2]

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        epsilon_init = 1.0
        self.epsilon = epsilon_init

        self.agent1 = MADDPGLearner(self.sess, 'agent_1')
        self.agent2 = MADDPGLearner(self.sess, 'agent_2')

        self.Saver = tf.train.Saver()
        self.train_mode = True
        # self.Summary, self.Merge = self.Make_Summary()

    def run_simulation(self):

        ###########################################################################################################

        run_episode = 10000
        test_episode = 100

        start_train_episode = 500

        save_interval = 5000

        #######################################################################################################

        step = 0

        # 게임 진행 반복문
        for episode in range(run_episode + test_episode):
            if episode == run_episode:
                self.train_mode = False

            # 유니티 환경 리셋 및 학습 모드 설정
            env_info = self.env.reset(train_mode=self.train_mode)

            # 첫번째 에이전트의 상태, episode_rewards, done 초기화
            state1 = env_info[self.brain_name1].vector_observations[0]
            episode_rewards1 = 0
            done1 = False

            # 두번째 에이전트의 상태, episode_rewards, done 초기화
            state2 = env_info[self.brain_name2].vector_observations[0]
            episode_rewards2 = 0
            done2 = False

            # 한 에피소드를 진행하는 반복문
            while not done1 or done2:
                step += 1

                # 액션 결정 및 유니티 환경에 액션 적용
                action1 = self.agent1.get_agent_action(state1, self.sess)
                action2 = self.agent2.get_agent_action(state2, self.sess)
                
                env_info = self.env.step(vector_action={
                    self.brain_name1: [action1],
                    self.brain_name2: [action2]
                })

                # 첫번째 에이전트에 대한 다음 상태, 보상, 게임 종료 정보 취득
                next_state1 = env_info[self.brain_name1].vector_observations[0]
                reward1 = env_info[self.brain_name1].rewards[0]
                episode_rewards1 += reward1
                done1 = env_info[self.brain_name1].local_done[0]

                # 두번째 에이전트에 대한 다음 상태, 보상, 게임 종료 정보 취득
                next_state2 = env_info[self.brain_name2].vector_observations[0]
                reward2 = env_info[self.brain_name2].rewards[0]
                episode_rewards2 += reward2
                done2 = env_info[self.brain_name2].local_done[0]

                # 학습 모드인 경우 리플레이 메모리에 데이터 저장
                if self.train_mode:
                    self.agent1.agent_memory.add((np.vstack([state1, state2]),
                                             np.vstack([action1, action2]),
                                             reward1,
                                             np.vstack([next_state1,
                                                        next_state2],
                                             False)))

                    if episode > start_train_episode:
                        # 학습 수행
                        train_agent(self.agent1, self.sess, [self.agent2])

            """"# 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수값 기록
            if episode % print_interval == 0 and episode != 0:
                print("step: {} / episode: {} / epsilon: {:.3f}".format(step, episode, agent.epsilon))
                print("reward1: {:.2f} / loss1: {:.4f} / reward2: {:.2f} / loss2: {:.4f}".format(
                    np.mean(rewards1), np.mean(losses1), np.mean(rewards2), np.mean(losses2)))
                print('------------------------------------------------------------')

                agent.Write_Summray(np.mean(rewards1), np.mean(losses1),
                                    np.mean(rewards2), np.mean(losses2), episode)
                rewards1 = []
                losses1 = []
                rewards2 = []
                losses2 = []"""

            # 네트워크 모델 저장
            if episode % save_interval == 0 and episode != 0:
                game = "Pong"
                date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
                save_path = "../saved_models/" + game + "/" + date_time + "_DQN"
                self.Saver.save(self.sess, save_path + "/model/model")
                print("Save Model {}".format(episode))

        self.env.close()


# Main 함수 -> 전체적으로 적대적인 DQN 알고리즘을 진행
if __name__ == '__main__':
    test = EnvironmentTest()
    test.run_simulation()
