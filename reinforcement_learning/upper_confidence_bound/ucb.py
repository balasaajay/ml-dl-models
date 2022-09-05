import matplotlib.pyplot as plt
import pandas as pd

# Import simulation dataset
# Reinforcement is mostly used for real time data
data = pd.read_csv("/Users/abalasa/github/balasaajay/ml-model-templates/reinforcement_learning/upper_confidence_bound/Ads_CTR_Optimisation.csv")

# Implement UCB algo
import math
N = 1000  # num of users or num of runs. Try to change this and see how algo behaves
d = 10 # num of ads
ads_selected = list()
num_of_selections = [0] * d   # (Ni(n))
sums_of_rewards = [0] * d     #  Ri(n)
total_reward = 0
for n in range(0, N):
  # start from the first ad
  ad = 0
  max_upper_bound = 0 
  for i in range(0, d):
    if num_of_selections[i] > 0:
      avg_reward = sums_of_rewards[i] / num_of_selections[i]
      delta_i = math.sqrt(1.5 * (math.log(n+1)/num_of_selections[i]))
      upper_bound = avg_reward + delta_i
    else:
      upper_bound = 1e400

    if upper_bound > max_upper_bound:
      max_upper_bound = upper_bound
      ad = i
    ads_selected.append(ad)
    num_of_selections[ad] = num_of_selections[ad] + 1
    reward = data.values[n, ad] # get the reward for the ad
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward += reward

# Visualize the models
# Ad vs number of times it was selected
plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times')
plt.show()
