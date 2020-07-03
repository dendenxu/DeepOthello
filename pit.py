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
class NNetPlayer:
    def __init__(self, game, args, chkpt):
        self.n = NNet(game)
        self.n.load_checkpoint(chkpt[0], chkpt[1])
        self.mcts = MCTS(game, self.n, args)

    def play(self, board):
        return np.argmax(self.mcts.getActionProb(board, temp=0))


chkpt1 = ('./temp', 'best.pth.tar')
args1 = dotdict({'numMCTSSims': 200, 'cpuct': 1.0})

n1p = NNetPlayer(g, args1, chkpt1).play

chkpt2 = ('./temp', 'best.pth.tar')
args2 = dotdict({'numMCTSSims': 500, 'cpuct': 1.0})

n2p = NNetPlayer(g, args2, chkpt2).play


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
