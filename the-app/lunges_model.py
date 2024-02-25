import numpy as np
import pandas as pd

'''
df = pd.read_csv("lunges.csv")

averages = df.mean(axis=1)
std_devs = df.std(axis=1)

print(averages)
print(std_devs)
'''

class exercise_model:
    def __init__(self, fileName) -> None:
        self.df = pd.read_csv(fileName)
        self.df = self.df.iloc[1:]
        self.averages = self.df.mean(axis=1)
        self.std_devs = self.df.std(axis=1)
    
    def getError(self,dfToCheck):
        error_rates = []
        for column in range(len(self.df.columns)):
            error_rate = self.std_devs.iloc[int(column)] / self.averages.iloc[int(column)]
            error_rates.append(error_rate)
        return error_rates
    
    
lunges_avg = exercise_model("lunges.csv")
print(lunges_avg.getError(lunges_avg))
