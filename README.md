# Reinforcement Learning

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

### 2.  Did you try any other changes (like adding layers or changing the epsilon value) that made things better or worse?
### 3.  Did your changes improve or degrade the model? How close did you get to a test run with 100% of the scores above 200?
### 4.  Based on what you observed, what conclusions can you draw about the different parameters and their values?
### 5.  What is the purpose of the epsilon value?
### 6.  Describe "Q-Learning".
