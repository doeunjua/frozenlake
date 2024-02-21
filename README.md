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
