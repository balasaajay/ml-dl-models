import matplotlib.pyplot as plt
import pandas as pd

# Import simulation dataset
# Reinforcement is mostly used for real time data
data = pd.read_csv("/Users/abalasa/github/balasaajay/ml-model-templates/reinforcement_learning/upper_confidence_bound/Ads_CTR_Optimisation.csv")


# Implement Thompson sampling algo
import random
N = 500 # num of users or num of runs. Try to change this and see how algo behaves
# started with 10000 but 500 also gave the same value and that is why we have 500 here
d = 10 # num of ads
ads_selected = list()
num_rewards_0 = [0] * d
num_rewards_1 = [0] * d
total_reward = 0
for n in range(0, N):
  ad = 0 # index of ad selected
  max_random = 0
  for i in range(0, d):
    random_beta = random.betavariate(num_rewards_0[i]+1, num_rewards_1[i]+1)
    if max_random < random_beta:
      ad = i
      max_random = random_beta
  
  ads_selected.append(ad)
  reward = data.values[n, ad]
  if reward == 1:
    num_rewards_1[ad] += 1
  else:
    num_rewards_0[ad] += 1
  total_reward += reward

# Visualize the models
# Ad vs number of times it was selected
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times')
plt.show()
