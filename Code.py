#================================================
#IMPORT LIBRARIES
#================================================
import math
import pandas as pd
import numpy as np
from scipy.stats import norm
import time
import random
import matplotlib.pyplot as plt
import statistics
%matplotlib inline

"""
#================================================
#INITIAL INPUTS
#================================================
S_ZERO = 100
STRIKE_E = 100
TIME_TO_EXPIRY = 1
VOLATILITY = 0.2
RISK_FREE_RATE = 0.05
N = 1
"""

#================================================
#BLACK SCHOLES
#================================================

def BINARY_BS_OPTION_VALUE(S_ZERO, STRIKE_E, VOLATILITY, RISK_FREE_RATE, TIME_TO_EXPIRY):
    
    #calculating D2
    D2 = (math.log(S_ZERO/STRIKE_E) + (RISK_FREE_RATE - 0.5*VOLATILITY**2) * \
          TIME_TO_EXPIRY)/ (VOLATILITY * math.sqrt(TIME_TO_EXPIRY))
    
    #present value discounting
    PRESENT_VALUE_DISC = math.e**(-RISK_FREE_RATE*TIME_TO_EXPIRY)
    
    #finding prices
    CALL_OPTION_VALUE = PRESENT_VALUE_DISC * norm.cdf(D2)
    PUT_OPTION_VALUE = PRESENT_VALUE_DISC * (1 - norm.cdf(D2))

    #printing results
    print('*** BLACK SCHOLES FORMULA ***')
    print('------------------------------')
    print('Call Value: ' + str(round(CALL_OPTION_VALUE,6)))
    print('Put Value: ' + str(round(PUT_OPTION_VALUE,6)))
    print('------------------------------')

#================================================
#MONTECARLO - Euler-Maruyama method
#================================================

def MONTECARLO_EULER_MARUYAMA_OPTION_VALUE (S_ZERO, STRIKE_E, VOLATILITY, RISK_FREE_RATE, \
                                            TIME_TO_EXPIRY, N_SIMULATIONS, n_SIMULATIONS):
    
    
    #quick function to calculate payoff (i.e. 0 or 1)
    def binary_call_payoff(STRIKE_E, S_T):
        if S_T >= STRIKE_E:
            return 1.0
        else:
            return 0.0
    
    LIST_CALL_OPTION_VALUE = []
    LIST_PUT_OPTION_VALUE = []
    
    for i in range(n_SIMULATIONS):
    
        #running simultaions
        payoffs = 0.0
        for i in range(N_SIMULATIONS):

            #initial settings to simulate the process
            dt = 0.001  # Time step.
            n = int(TIME_TO_EXPIRY / dt)  # Number of time steps
            t = np.linspace(0, TIME_TO_EXPIRY, n)  # Vector of times.

            #creating an array for the values of the underlying (s)
            x = np.zeros(n)
            x[0]= S_ZERO

            #calculating the value of the underlying (s) at each time-step
            for i in range(n - 1):
                x[i + 1] = x[i] * (1 + RISK_FREE_RATE * dt + VOLATILITY * np.sqrt(dt) * np.random.randn() )

            #returning the last element of the array
            S_T = x[-1] 

            #adding 0 or 1 for each iteration
            payoffs += binary_call_payoff(STRIKE_E, S_T)


        #present value discounting
        PRESENT_VALUE_DISC = math.e**(-RISK_FREE_RATE*TIME_TO_EXPIRY)

        #finding prices (i.e. discounting the average payoff)
        CALL_OPTION_VALUE = PRESENT_VALUE_DISC * (payoffs / float(N_SIMULATIONS))
        PUT_OPTION_VALUE = PRESENT_VALUE_DISC * (1-(payoffs / float(N_SIMULATIONS)))
    
        LIST_CALL_OPTION_VALUE.append(CALL_OPTION_VALUE)
        LIST_PUT_OPTION_VALUE.append(PUT_OPTION_VALUE)
    
    print('*** MONTECARLO - EULER MARUYAMA METHOD ***')
    print('------------------------------')
    print('N = ' + str(N_SIMULATIONS))
    print('n = ' + str(n_SIMULATIONS))
    print('Call Value: ' + str(round(statistics.mean(LIST_CALL_OPTION_VALUE),6)))
    print('Put Value: ' + str(round(statistics.mean(LIST_PUT_OPTION_VALUE),6)))
    print('------------------------------')

    


#================================================
#MONTECARLO - Milstein Correction
#================================================

def MONTECARLO_MILSTEIN_CORRECTION_OPTION_VALUE (S_ZERO, STRIKE_E, VOLATILITY, RISK_FREE_RATE, TIME_TO_EXPIRY, \
                                                 N_SIMULATIONS, n_SIMULATIONS):
    
    
    #quick function to calculate payoff (i.e. 0 or 1)
    def binary_call_payoff(STRIKE_E, S_T):
        if S_T >= STRIKE_E:
            return 1.0
        else:
            return 0.0
    
    LIST_CALL_OPTION_VALUE = []
    LIST_PUT_OPTION_VALUE = []
    
    for i in range(n_SIMULATIONS):
    
    
        #running simultaions
        payoffs = 0.0
        for i in range(N_SIMULATIONS):

            #initial settings to simulate the process
            dt = 0.001  # Time step.
            n = int(TIME_TO_EXPIRY / dt)  # Number of time steps
            t = np.linspace(0, TIME_TO_EXPIRY, n)  # Vector of times.
            MILSTEIN_CORRECTION = 0.5*VOLATILITY**2*(np.random.randn()**2 - 1)*dt

            #creating an array for the values of the underlying (s)
            x = np.zeros(n)
            x[0]= S_ZERO

            #calculating the value of the underlying (s) at each time-step
            for i in range(n - 1):
                x[i + 1] = x[i] * (1 + RISK_FREE_RATE * dt + VOLATILITY * np.sqrt(dt) * np.random.randn() \
                                   + MILSTEIN_CORRECTION)

            #returning the last element of the array
            S_T = x[-1] 

            #adding 0 or 1 for each iteration
            payoffs += binary_call_payoff(STRIKE_E, S_T)


        #present value discounting
        PRESENT_VALUE_DISC = math.e**(-RISK_FREE_RATE*TIME_TO_EXPIRY)

        #finding prices (i.e. discounting the average payoff)
        CALL_OPTION_VALUE = PRESENT_VALUE_DISC * (payoffs / float(N_SIMULATIONS))
        PUT_OPTION_VALUE = PRESENT_VALUE_DISC * (1-(payoffs / float(N_SIMULATIONS)))
    
        LIST_CALL_OPTION_VALUE.append(CALL_OPTION_VALUE)
        LIST_PUT_OPTION_VALUE.append(PUT_OPTION_VALUE)
    
    
    print('*** MONTECARLO - MILSTEIN CORRECTION ***')
    print('------------------------------')
    print('N = ' + str(N_SIMULATIONS))
    print('n = ' + str(n_SIMULATIONS))
    print('Call Value: ' + str(round(statistics.mean(LIST_CALL_OPTION_VALUE),6)))
    print('Put Value: ' + str(round(statistics.mean(LIST_PUT_OPTION_VALUE),6)))
    print('------------------------------')