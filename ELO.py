from ELO_Rank import rank
import pandas as pd
import numpy as np


##Elo rank system 
#create database/ dictionary of teams & elo
def main():
#call a function that creates a team profile {name, ELO}
    for t in df:
        if ('teamA' in t for t in teams):
            pass
        else:
            teams[t] = 1000
    for t in df:
        if ('teamB' in t for t in teams):
           pass
        else:
            teams[t] = 1000

    while (t < T): 
#call a function that takes game data & returns a new rank {name A, name B, score A, score B}
        NELO = rank(teams[df.teamA[t]],df.S_A[t], teams[df.teamB[t]],df.S_A[t])
#append teams with New elo
        teams.update({df.teamA[t] : NELO[0,1]})
        teams.update({df.teamB[t] : NELO[0,2]})
        #break while loop
        t = t+1
#append the current team profile with new ELO



#check data types: 'df.dtypes' names as str, numeric as int
df = pd.read_csv("elo.csv")
#teams should track old rank using [elo,elo,elo]
teams = {'teams' : 0} 
t = 0
for x in teams:
    T = T + 1


if __name__ == '__main__':
    main()