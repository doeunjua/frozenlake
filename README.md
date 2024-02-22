# frozenlake
## **학습목표** 
## 1. gym에 대해 알아보기
## 2. Q-table로 frozenlake 구현하기
## 3. Q-table과 dqn뭐가 더 적합할까 생각해보기
## 4. DQN으로 frozenlake 구현하기


# **1.gym에 대해 알아보기**
## [1] gym 사이트 활용하기

openAI 에서 여러가지 게임들을 통해 강화학습을 테스트 할 수 있는 Gym 이라는 환경을 제공해줍니다. 아래 링크를 걸어놨는데 일단 여기 들어가보세요.
[gym사이트](https://www.gymlibrary.dev/)

사이트에 들어가서 왼쪽을보면 아래 첨부한 이미지를 찾을 수 있습니다. 이게 gym에서 제공하는 환경들입니다.

<img width="188" alt="image" src="https://github.com/doeunjua/frozenlake/assets/122878319/c5980f5c-794e-4087-9b25-70caeb4eee02">



이중에서 저희가 목요일날 dqn으로 구현해 볼 것은 Toy text에 들어가서 네번째에 있는 frozenlake입니다.


[frozenlake](https://www.gymlibrary.dev/environments/toy_text/frozen_lake/)


제가 어떻게 해결해야할 문제를 분석하는지는 목요일날 알려드리겠습니다.
<img width="345" alt="image" src="https://github.com/doeunjua/frozenlake/assets/122878319/bacc7f53-ab28-4816-927a-6ec7ac6aacd3">

<img width="134" alt="image" src="https://github.com/doeunjua/frozenlake/assets/122878319/277d3c0f-5259-4366-9233-ee7ea0ff8d44">


frozenlake는 얼어붙은(F) 호수 위를 걸어서 홀(H)에 빠지지 않고 시작 지점(S)에서 목표 지점(G)까지 얼어붙은 호수를 건너는 게임입니다. 그림에서 인형이 처음 0,0 위치에서부터 시작해서 선물 그림이 있는 3,3 위치에 도착하는것을 목표로합니다. 바로 위의 사진에 보상이 어떻게 주어지는지 첨부해놓았습니다. 선물이 있는 곳까지 잘 도착했으면 보상 1을 받고 게임이 끝나고 만약에 중간에 hole에 빠졌으면 보상을 받지않고(보상:0) 게임이 끝납니다. 여기서는 사용자 마음대로 맵을 바꿀수 있는데 제가 첨부해놓은 그림대로 맵을 만들면 아래와 같이 만들 수 있습니다.
```python
custom_map = [
    'FFFF',
    'FHFH',
    'FFFH',
    'HFFG'
]
```
행동은 4가지가 있고 아래와 같습니다.
### 0: LEFT
### 1: DOWN
### 2: RIGHT
### 3: UP
그러면 이제 구현시작해볼게요. 우선 gym을 import해야하는데 먼저 깔고 시작해야합니다.
## **숙제1. gym 깔아오기**
```python
pip install gym
```
깔았으면 import해서 사용할 수 있습니다.
```python
import gym
```
```python
env=gym.make("FrozenLake-v1")
```
이 코드는 make라는 함수를 써서 강화학습 환경 frozenlake를 생성하는 코드입니다.

`env.observation_space.n`을 하면 관찰할 공간의 수 그다음 `env.action_space.n`을 하면 액션의 수를 알 수 있습니다.
<img width="476" alt="image" src="https://github.com/doeunjua/frozenlake/assets/122878319/87afe2d0-11dc-48c4-8c15-21be69c6a057">
여기서 관찰 공간 0부터 15까지 16개 그다음 액션 위, 아래, 왼쪽,오른쪽 해서 4개 이렇게 나오네요!

제가 뒤에 코드를 드릴껀데 거기 보면`env.reset()`이라는 줄 이 나옵니다. 이것은 enviromant를 기본 상태로 재설정하는 코드입니다.
<img width="290" alt="image" src="https://github.com/doeunjua/frozenlake/assets/122878319/3c774799-c3c3-4977-aacd-e5716af913ce">

돌려보니 이런꼴로 나오네요 그래서 초기 상태 맨 처음 인형이 있는 위치가 상태 0번인데 거기서 다시 시작하겠다라고 설정해주려면 `env.reset()[0]`을 해서 0만 추출해줍니다. 아니면 이 문제에서는 `state=0`으로 지정해주고 시작해도 똑같습니다. 하지만 상태가 이렇게 딱 정수로 시작하지 않는 경우에는 `env.reset()`을 써줘야겠죠???

<img width="658" alt="image" src="https://github.com/doeunjua/frozenlake/assets/122878319/58271d7c-3f66-4742-b0cf-ca2f5028ca1c">


그다음 `env.action_space.sample()`을 해주면 random action을 가져올 수 있습니다.

<img width="660" alt="image" src="https://github.com/doeunjua/frozenlake/assets/122878319/f25edffd-20e3-4529-a291-8e030e6f21d4">


`env.step(action)`을 해주면 action실행 후 상태 보상 행동이 끝났는지 여부 등 관련정보를 알 수 있습니다. 여기서는 인형이 젤 처음 위치해 있는 (0,0)이 상태 0입니다. 여기서 시작해서 action 3하고 step을 통해 확인해보면 0이 나옵니다. 젤 윗줄에서 위로 움직인거니까 next_state가 그대로 0인것을 확인할 수 있죠!

<img width="683" alt="image" src="https://github.com/doeunjua/frozenlake/assets/122878319/6d96e6b2-3f33-4964-aac2-a9d1aef4b2f9">

만약에 다시 돌려서 action이 2가 나오고 `env.step(action)`해주면 다음상태가 한칸 오른쪽인 1인것을 알 수 있습니다.
여기까지 기본적인 함수와 파라미터 확인 방법에 대해 설명드렸습니다. 
# **2. Q-table로 frozenlake 구현하기**
## **숙제2. 코드를 마저 완성해서 Q-table확인하기(코드1,코드2에 맞는 코드 채우시면 됩니다)**
```python
import numpy as np
import gym
import random
import matplotlib.pyplot as plt

custom_map = [
    'FFFF',
    'FHFH',
    'FFFH',
    'HFFG'
]

env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False, render_mode='human')
state_size = env.observation_space.n
action_size = env.action_space.n

Q = np.zeros((state_size, action_size))


alpha = 0.8  # Learning rate
gamma = 0.99  # 할인율
epsilon = 1.0  
episodes = 1000


rewards_per_episode = []

rewards = []

for episode in range(episodes):
  state = env.reset()[0]

  for _ in range(1000):
    #코드 1. 입실론 활용해서 액션선택하기

    
    next_state, reward, done, _,_ = env.step(action)

    #코드 2. Q값 업데이트 공식사용해서 Q테이블 업데이트 후 다음상태 바꿔주기

    if done:#홀에빠지거나 선물에 도착하면 게임이 끝남 그러면 done이 1로 바뀜
      rewards.append(reward)
      epsilon -= 0.001
      break

print(Q)

print(np.average(rewards))
```
<img width="589" alt="image" src="https://github.com/doeunjua/frozenlake/assets/122878319/2fd0c030-8cf4-4366-a028-e5802faab7e5">

저는 이런식으로 Q table과 평균보상이 나오네요. 아마 여러분들도 코드를 잘 짰다면 완전히 똑같지는 않더라도 거의 유사한 테이블이 나올겁니다.

# **3.Q-table과 dqn뭐가 더 적합할까 생각해보기**

퀴즈. frozenlake문제에서는 qlearning과 dqn중 뭐가 더 적합할까요?
Q-table과 dqn중 뭐가 더 적합할지에 대한 나름의 기준을 gpt에게 물어보니 아래의 그림에서 볼 수 있듯이 몇가지를 설명해줍니다. 
## **숙제3. frozenlake문제는 Q-table과 dqn중 뭐가 더 적합할지 첨부한 그림에 나온 기준을 바탕으로 생각해오기**
<img width="473" alt="image" src="https://github.com/doeunjua/frozenlake/assets/122878319/a86b243a-cfbc-4eaa-bf6e-44dc48f5585f">

# **4. DQN으로 frozenlake 구현하기**

수업시간에 같이 해봅시다.
```python
import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt



custom_map = [
    'FFFF',
    'FHFH',
    'FFFH',
    'HFFG'
]

env = gym.make("FrozenLake-v1", desc=custom_map, is_slippery=False)
state_size = env.observation_space.n
action_size = env.action_space.n


class DQNAgent:
    def __init__(self, state_size, action_size): #하이퍼 파라미터값 설정 여러분들 맘대로 설정해보셔도 돼요요

        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        #deque는 양방향 큐로, maxlen으로 최대 길이를 설정함으로써 고정된 크기의 메모리를 유지할 수 있습니다.
        self.gamma = 0.95    
        self.epsilon = 1.0  
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model() #클래스 호출할 때 모델을 만들어 주는 코드에요 재필이 오빠 수업때 DQN(1, 4, 0.001)을 호출하는 부분 기억 나시나요? 이 코드는 그것의 역할을 해요. 

    def _build_model(self): # 재필이 오빠가 할때는 layers.Dense해서 신경망 만들고 call함수에 넣어줘서 순전파 과정 거쳤죠? 근데 sequential쓰면 알아서 다 해줍니다.
        
        model = Sequential()
        model.add(Dense(16, input_dim=self.state_size, activation='relu'))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        #손실 함수로는 평균 제곱 오차(MSE: Mean Squared Error)를 사용하며, 이는 에이전트의 예측과 실제 보상 사이의 차이를 최소화하는 데 사용됩니다.
        return model

    def remember(self, state, action, reward, next_state, done): #메모리에 저장하는 함수
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state): #입실론 그리디 방법을 사용해서 액션을 선택하는 함수
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        #random.randrange(self.action_size) 또는 randint써도됨
        act_values = self.model.predict(state)
        #self.model.predict(state)이렇게 predict함수 안에 상태를 넣으면 주어진 상태에 대해 에이전트의 신경망 모델을 사용하여 각 행동에 대한 예상 가치(Q-값)를 계산해서 결과로 줍니
        return np.argmax(act_values[0])  #np.argmax(act_values[0])를 사용하여 가장 높은 예상 가치를 가진 행동의 인덱스를 선택할 수 있도록 해줍니다

    def replay(self, batch_size):
        #replay 메서드는 DQN 알고리즘의 핵심이에요. 이전에 memory에 저장했던것을 활용해서 경험했던것을 replay해주는 코드에요
        #에이전트가 과거의 경험(상태, 행동, 보상, 다음 상태의 세트)을 다시 학습함으로써 학습 과정을 안정시키고 효율성을 높이는 데 도움을 줄 수 있어요
        minibatch = random.sample(self.memory, batch_size)#먼저 에이전트의 메모리(self.memory)에서 임의로 batch_size만큼의 경험을 샘플링합니다
        for state, action, reward, next_state, done in minibatch:
        #각 반복에서 한 경험의 구성 요소(상태, 행동, 보상, 다음 상태, 종료 여부)를 가져옵니다
            target = reward
#목표 Q-값(target)을 계산합니다. 에피소드가 끝나지 않았다면(not done), 목표 값은 현재 보상과 다음 상태에서 가능한 최대 Q-값의 할인된 합으로 설정됩니다. np.amax배우셨나요? np.argmax가 최대 Q값을 가지는 행동을 선택하기 위해 인덱스를 뽑는거라면 np.amax는 다음 상태에서 예측된 모든 가능한 행동 가치 중 최댓값을 선택합니다.
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
#현재 상태에 대해 모델이 예측한 행동 가치들을 가져옵니다(target_f). 그런 다음, 실제로 선택된 행동(action)에 해당하는 가치를 위에서 계산한 목표 Q-값으로 업데이트합니다. 
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)# 우리가 target값 업데이트 했죠 그러면 이제 이 값을 사용하여 신경망의 가중치를 업데이트 해서 에이전트는 장기적인 보상을 최대화하는 행동을 학습하게 됩니다. 이런 방식으로 경험을 replay해서 가중치를 업데이트 해준다는 것 이해가나요? 이해 안간다면 그냥 받아들이는 것을 추천드립니다.

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        #스텝이 지날수록 입실론값 감소해서 학습할수록 점점더 탐험보다는 활용을 더 높게 해줄게요 Qtable에서는 0.001씩 빼주는걸로 값 수정을 했었는데 여기서는 0.995씩 곱해서 감소시켰어요 편하신대로 하시면 돼요

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def one_hot_state(state): #one hot encoding방식을 위한 함수입니다다
    one_hot = np.zeros(state_size)
    one_hot[state] = 1
    return np.reshape(one_hot, [1, state_size])#np.reshape배우셨나요? 이게 dqn구현 연습하면서 되게 중요할 것 같아요 모르신다면 직접 구글링해서 사용방법을 익히는 것을 추천드립니다. 간단하게 설명드리자면 one hot결과를 [1, state_size] 형태의 2차원 배열로 재구성해줘요. 예를들어 볼게요 state_size가 4라고 가정하고 특정 state가 2인 경우를 생각해볼게요. 이때 one_hot 벡터는 [0, 0, 1, 0]로 설정됩니다. 여기서 np.reshape(one_hot, [1, state_size])이 코드를 적용해주면 [[0,0,1,0]]이렇게 2차원으로 구성해주는거죠. 여러 dqn코드 짜다보면 shape오류가 많이 나요. 저는 그랬어요... 그래서 reshape를 잘 알아두는 것을 추천드릴게

agent = DQNAgent(state_size, action_size)


import matplotlib.pyplot as plt

EPISODES = 1000
BATCH_SIZE = 32  


rewards_per_episode = []

for e in range(EPISODES):
    state = env.reset()[0]
    #env.reset()의 결과가 다들 뭔지 기억나시나요? 튜플형태였는데 제가 2주전에 파이썬에서는 튜플을 인덱스로 접근가능하다는 것을 알려드렸죠? 첫번째 요소를 가져오게 할 수 있는 코드에요
    #state,_=env.reset()이렇게 해줘도 같은 결과가 state에 들어가요
    state = one_hot_state(state)
    done = False
    total_reward = 0
    
    while not done:
       
        action = agent.act(state)
        next_state, reward, done, _ ,_= env.step(action)
        next_state = one_hot_state(next_state)

        #보상체계를 수정해보자
        
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

        if len(agent.memory) > BATCH_SIZE:
            agent.replay(BATCH_SIZE)

    if agent.epsilon > agent.epsilon_min:#이거는 경험의 최솟값을 보장해주겠다는 거에요. 이렇게 해도되고 아니면 위에서 act함수에서 max(self.epsilon, self.epsilon_min)이렇게 해줘도 돼요
        agent.epsilon *= agent.epsilon_decay
    print(f"Episode: {e+1}/{EPISODES}, Total Reward: {total_reward}, Final Reward: {reward}")

    
    rewards_per_episode.append(total_reward)
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Rewards per Episode Over Time')
plt.show()
```
