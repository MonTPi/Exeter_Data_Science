def rank(eloA, S_A, eloB, S_B):
    #transform rating
    R_a = 10**(eloA/400)
    R_b = 10**(eloB/400)
    #expected score
    E_a = R_a / (R_a + R_b)
    E_b = R_b / (R_a + R_b)

    if (S_A > S_B):
        s_a = 1
        s_b = 0
    elif (S_A < S_B):
        s_a = 0
        s_b = 1
    else :
        s_a = 0.5
        s_b = 0.5

    eloA = eloA + (K * (s_a - E_a))
    eloB = eloB + (K * (s_b - E_b))

    x = [eloA,eloB]

    return x
    


K = 20


if __name__ == '__main__':
    rank()
