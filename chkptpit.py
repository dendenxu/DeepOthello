import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet


import numpy as np
from utils import *

import os
import logging

import coloredlogs
log = logging.getLogger(__name__)
coloredlogs.install("INFO")
"""
use this script to play any two agents against each other, or play manually with
any agent.
"""


# nnet players
class NNetPlayer:
    def __init__(self, game, args, chkpt):
        self.n = NNet(game)
        self.n.load_checkpoint(chkpt[0], chkpt[1])
        self.mcts = MCTS(game, self.n, args)

    def play(self, board):
        return np.argmax(self.mcts.getActionProb(board, temp=0))


def chkptpit(game, players, max_iter=1000, num_games=20):


    for i in range(1, max_iter):
        path = f"temp/checkpoint_{i}.pth.tar"
        if not os.path.exists(path):
            log.info("All iterations are done now.")
            break
        for desc2, player2 in players.items():
            # I personally think we should clear the tree after pitting with a particular player
            chkpt = path.split("/")
            args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
            player1 = NNetPlayer(game, args, chkpt).play
            desc1 = f"nnet-50_{i}"
            arena = Arena.Arena(player1, player2, game, display=OthelloGame.display)
            log.info(f"{desc1} vs {desc2}")
            log.info(arena.playGames(num_games))

def main():
    g = OthelloGame(6)

    # all players
    rp = RandomPlayer(g).play
    gp = GreedyOthelloPlayer(g).play
    hp = HumanOthelloPlayer(g).play
    sp = SimplePlayer(g).play
    scp = SingleCorePlayer(g).play
    players = {
        "random": rp,
        "greedy": gp,
        "alphabeta-simple": sp,
        "alphabeta-strategy": scp,
    }
    chkptpit(g, players)

if __name__ == "__main__":
    main()
