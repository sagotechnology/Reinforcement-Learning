# Reinforcement Learning

![caption](https://j.gifs.com/Dq1Kyx.gif)

### 1.  What parameters did you change, and what values did you use?

```
class DQN:
    def __init__(self, env):

        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.counter = 0

        #######################
        # Change these parameters to improve performance
        self.density_first_layer = 16
        self.density_second_layer = 32
        self.density_third_layer = 64
        self.num_epochs = 1
        self.batch_size = 64
        self.epsilon_min = 0.01

        # epsilon will randomly choose the next action as either
        # a random action, or the highest scoring predicted action
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.gamma = 0.99

        # Learning rate
        self.lr = 0.001

        #######################

        self.rewards_list = []

        self.replay_memory_buffer = deque(maxlen=500000)
        self.num_action_space = self.action_space.n
        self.num_observation_space = env.observation_space.shape[0]


        self.model = self.initialize_model()

    def initialize_model(self):
        model = Sequential()
        model.add(Dense(self.density_first_layer, input_dim=self.num_observation_space, activation=relu))
        model.add(Dense(self.density_second_layer, activation=relu))
        model.add(Dense(self.density_third_layer, activation=relu))
        model.add(Dense(self.num_action_space, activation=linear))

        # Compile the model
        model.compile(loss=mean_squared_error,optimizer=Adam(lr=self.lr))
        print(model.summary())
        return model
```

For my final model, I left the input layer the same, 16.  I then added a layer and set the number of nodes to 32 and 64 for layers two and three.  I used the original batch size, epsilon min, epsilon decay, and learning rate of 64, 0.01, 0.995, and 0.001, respectively. With these parameters, it took about one hour to train the model.  The average reward was 206 during testing.  

### 2.  Did you try any other changes (like adding layers or changing the epsilon value) that made things better or worse?  
During my training, adding a third layer improved the model.  Decreasing the epsilon decay to 0.9 produced a horrendous model.  


### 3.  Did your changes improve or degrade the model? How close did you get to a test run with 100% of the scores above 200?  
Overall, my changes improved the model.  I was able to achieve 71% above 200 during testing.  

### 4.  Based on what you observed, what conclusions can you draw about the different parameters and their values?  

More nodes didn't always produce a better performance during training.  Decreasing the epsilon decay to 0.9 dramatically reduced the performance.  It was essential to select an optimum learning rate and batch size.  Also, adding a third layer improved the performance.  

### 5.  What is the purpose of the epsilon value?  
The epsilon value is the probability of choosing a random state rather than the highest Q value.  It allows the model to explore the state space.  Epsilon decay should produce a better model when used effectively.   

### 6.  Describe "Q-Learning".  
Q-learning is a type of reinforcement learning that seeks to find the best action to take given the current state.  It learns from actions that are outside the current policy; therefore, it is within the off-policy category. In a nutshell, Q-learning seeks to learn a policy that maximizes the total reward.   

![Screen Shot 2021-07-18 at 10 15 07 PM](https://user-images.githubusercontent.com/85311683/126106509-f62e3c55-0652-41fd-9f91-ad8a4f82e753.png)

Above is a diagram that represents reinforcement learning.  I will define some of the terminologies.  
1.  Agent - the learner and the decision-maker.
2.  Environment - where the agent learns and decides what actions to perform.
3.  Action - a set of actions that the agent can perform.
4.  State - the state of the agent in the environment.
5.  Reward - for each action selected by the agent, the environment provides a reward.  Usually a scalar value. 

In summary, the environment provides several states to choose from.  The agent then takes action and is rewarded based on the decision.  The agent is trained to maximize its reward.

https://sgomez-w251-hw11.s3.us-west-1.amazonaws.com/testing+run0.mp4
