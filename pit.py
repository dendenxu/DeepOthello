import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

import logging

import coloredlogs
log = logging.getLogger(__name__)
"""
use this script to play any two agents against each other, or play manually with
any agent.
"""


g = OthelloGame(6)

# all players
rp = RandomPlayer(g).play
gp = GreedyOthelloPlayer(g).play
hp = HumanOthelloPlayer(g).play
sp = SimplePlayer(g).play
scp = SingleCorePlayer(g).play



# nnet players
n1 = NNet(g)
n1.load_checkpoint('./temp','best.pth.tar')
args1 = dotdict({'numMCTSSims': 200, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))


n2 = NNet(g)
n2.load_checkpoint('./temp', 'best.pth.tar')
args2 = dotdict({'numMCTSSims': 500, 'cpuct': 1.0})
mcts2 = MCTS(g, n2, args2)
n2p = lambda x: np.argmax(mcts2.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, scp, g, display=OthelloGame.display)

print(arena.playGames(2, verbose=True))

# players = {
#     "random": rp,
#     "greedy": gp,
#     "alphabeta-simple": sp,
#     "alphabeta-strategy": scp,
#     "nnet-200": n1p,
#     "nnet-500": n2p,
# }

# for desc1, player1 in players.items():
#     for desc2, player2 in players.items():
#         arena = Arena.Arena(player1, player2, g, display=OthelloGame.display)
#         print(f"{desc1} vs {desc2}")
#         print(arena.playGames(2, display_result=True))
