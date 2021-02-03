# Deep Q-Learning

For this exercise, I implemented Deep Q-Learning to the OpenAI Gym CarRacing-v0 environment. The agent learned through experience using only pixels as input. Similar techniques used in [Mnih et al., Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) were implemented here such as the use of a replay buffer and two separate networks for the policy. Other techniques used included use of epsilon decay and polyak averaging between the target and policy network. Below is the epsilon decay schedule used. 

![epsilon_decay](https://github.com/erasromani/deep-q-learning/blob/main/images/epsilon_decay.png)

For the policy, a standard four layer convolutional neural network was used followed by two fully connected layers. A grid search approach was used for hyperparameter tuning. I realized that the system was very sensitive to the hyperparameters and it was difficult to get stable training. Training loss did not appear to be a good metric for determining whether or not the system is learning. I therefore had to rely on the reward as an indicator. Below is a loss and reward per episode throughout training.

![loss](https://github.com/erasromani/deep-q-learning/blob/main/images/loss.png)

![reward_history](https://github.com/erasromani/deep-q-learning/blob/main/images/reward_history.png)

I evaluated the system performance using the weights obtained from different points in time throughout training. As you can see from the figure below, the best performance is attained after 500 training episodes. After 500 episodes, the performance starts to drop.

![test_performance](https://github.com/erasromani/deep-q-learning/blob/main/images/test_performance.png)

Below are screenshots of the agent during testing for reference.

![screenshots](https://github.com/erasromani/deep-q-learning/blob/main/images/screenshots.PNG)
