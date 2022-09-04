import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import Dataset
# First row is not header in the dataset
data = pd.read_csv("/Users/abalasa/github/balasaajay/ml-model-templates/association_rule_learning/apriori/Market_Basket_Optimisation.csv", header=None)
transactions = list()
# print(data)
for i in range(0, 7501):
  transactions.append([str(data.values[i, j]) for j in range(0,20)]) # list of lists

# Train the apriori model
from apyori import apriori
rules = apriori(transactions, 
        min_support= 0.003, # minimum number of transactions the 
                                  # product has to appear - 3/day => 21 per week; 
                                  # support=transactions per week/total number of transactions
        min_confidence=0.2,  # Rule to be correct 20% of the time. Try different vals
        min_lift = 3,        # measures the relevance of a rule. good lift is atleast 3
        min_length = 2,      # min and max lenghts define how many products should be part of a rule
        max_length = 2
        )

results = list(rules)

## Convert results to Pandas DF
def inspect(results):
    left         = [tuple(result[2][0][0])[0] for result in results]
    right         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(left, right, supports))
results_df = pd.DataFrame(inspect(results), columns = ['Prod 1', 'Prod 2', 'Support'])

# print(results_df)

# Print in descending order of lift
print(results_df.nlargest(n=10, columns='Support'))