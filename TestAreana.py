
from ELO_Rank import rank
import pandas as pd
df = pd.read_csv("elo.csv")
df = df.replace(u'\xa0', u'', regex=True)


r = rank(1000,4,1000,df.S_A[0])

print(r)

from ELO_Rank import rank
import pandas as pd
import numpy as np

#check data types: 'df.dtypes' names as str.
#numeric as int & will need to 'pd.read_csv('elo.csv', index_col=0)' when data not format as table.
#remove space after names in data for easy dict
df = pd.read_csv("elo.csv")
df = df.replace(u'\xa0', u'', regex=True)

#teams should track old rank using [elo,elo,elo]
teams = {} 
t = 0
i = 0
#find df shape & return length as var T
T = df.shape[0]
##Elo rank system 
#create database/ dictionary of teams & elo

#call a function that creates a team profile {name, ELO}
for i in df.teamA[:]:
    if (i in teams.keys()):
        pass
    else:
        teams[i] = 1000
for i in df.teamB[:]:
    if (i in teams.keys()):
        pass
    else:
        teams[i] = 1000
while t < T: 
#call a function that takes game data & returns a new rank {name A, name B, score A, score B}
    NELO = rank(teams[df.teamA[t]],df.S_A[t], teams[df.teamB[t]],df.S_B[t])
#append teams with New elo
    teams.update({df.teamA[t] : NELO[0]})
    teams.update({df.teamB[t] : NELO[1]})
    #break while loop
    t = t+1

Rank = pd.DataFrame.from_dict([teams])
Rank = Rank.melt(id_vars = [], value_vars=Rank.columns.values, var_name='Team', value_name='ELO')
Rank = Rank.sort_values(['ELO'], ascending=[False])
print(Rank)
#append the current team profile with new ELO



