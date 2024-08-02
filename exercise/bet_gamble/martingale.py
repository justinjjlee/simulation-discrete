# %%
import random
import numpy as np
import pandas as pd

import seaborn as sns
'''
    Martingale Betting System
'''

class martingale:
    def __init__(self, DD, alpha=2, d=1):
        # Initial settings

        # Total budget
        self.DD = DD
        # Multiplying factor
        self.alpha = alpha
        #   Bet value - this to be updated while playing game
        self.d = d

        # Series outcome
        #   Series for total value
        self.D = [self.DD]
        #   Series for betting line
        self.b = [self.d]
        #   Series for probability outcome
        self.p = [0]

    def play(self):
        '''
            Draw the outcome:
                In this exmaple, I use equal probability

            One can develop further if needing to use different probabilty selection
        '''
        outcome = random.choice([-1, 1])
        # Outcome in level
        # If wanting to choose probability, changes required for other functions
        return outcome

    def returns(self, bet_i, outcome_lvl):
        if outcome_lvl >= 0:
            # Winning argument, the outcome
            return bet_i * self.alpha
        else:
            # Losing round, the outcome
            return bet_i * outcome_lvl
    
    def game(self):
        '''
            Play game
        '''
        # Keep playing game until total value you have is greater than betting requirement
        while self.D[-1] >= self.d:
            # Play round
            outcome = self.play()
            self.p.append(outcome)
            # Update returns
            returns = self.returns(self.d, outcome)
            # Update total value of your asset
            self.D.append(self.D[-1] + returns)
            # Update betting line for the next round
            self.d = self.d * self.alpha
            self.b.append(self.d)
# %% Simulation 

# Total iteration simulation
N_iter = 5_000
SIM = []
for iter in np.arange(0, N_iter):
    # Set game
    round_i = martingale(DD = 50_000, alpha = 2, d = 1)
    # Play game
    round_i.game()

    # Save as dataframe
    SIM.append(
        pd.DataFrame(
            {
                "Bet":round_i.b,
                "Asset":round_i.D,
                "p-outcome":round_i.p
            }
        )
    )