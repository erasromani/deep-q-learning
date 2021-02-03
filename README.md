# Deep Q-Learning

For this exercise, I implemented Deep Q-Learning to the OpenAI Gym CarRacing-v0 environment. The agent learned through experience using only pixels as input. Similar techniques used in Mnih et al., Playing Atari with Deep Reinforcement Learning were implemented here such as the use of a replay buffer and two separate networks for the policy. Other techniques used included use of epsilon decay and polyak averaging between the target and policy network. Below is the epsilon decay schedule used. 

For the policy, a standard four layer convolutional neural network was used followed by two fully connected layers. A grid search approach was used for hyperparameter tuning. I realized that the system was very sensitive to the hyperparameters and it was difficult to get stable training. Training loss did not appear to be a good metric for determining whether or not the system is learning. I therefore had to rely on the reward as an indicator. Below is a loss and reward per episode throughout training.


I evaluated the system performance using the weights obtained from different points in time throughout training. As you can see from the figure below, the best performance is attained after 500 training episodes. After 500 episodes, the performance starts to drop.

Below are screenshots of the agent during testing for reference.
