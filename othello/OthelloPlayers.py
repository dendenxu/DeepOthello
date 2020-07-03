import numpy as np


class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a


class HumanOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        # display(board)
        valid = self.game.getValidMoves(board, 1)
        for i in range(len(valid)):
            if valid[i]:
                print("[", int(i/self.game.n), int(i%self.game.n), end="] ")
        while True:
            input_move = input()
            input_a = input_move.split(" ")
            if len(input_a) == 2:
                try:
                    x,y = [int(i) for i in input_a]
                    if ((0 <= x) and (x < self.game.n) and (0 <= y) and (y < self.game.n)) or \
                            ((x == self.game.n) and (y == 0)):
                        a = self.game.n * x + y if x != -1 else self.game.n ** 2
                        if valid[a]:
                            break
                except ValueError:
                    # Input needs to be an integer
                    'Invalid integer'
            print('Invalid move')
        return a


class GreedyOthelloPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        valids = self.game.getValidMoves(board, 1)
        # print(board, type(board))
        # print(valids, type(board))
        candidates = []
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, 1, a)
            score = self.game.getScore(nextBoard, 1)
            candidates += [(-score, a)]
        candidates.sort()
        return candidates[0][1]

class SimplePlayer:
    '''An AIPlayer that gives moves according to current board status'''

    def __init__(self, game, big_val=1e10, small_val=-1e10, max_depth=6, max_width=12):
        self.game = game
        self.color = 1  # Defining the color of the current player, use 'X' or 'O'
        self.big_val = big_val  # A value big enough for beta initialization
        self.small_val = small_val  # A value small enough for alpha initialization
        self.depth = max_depth  # Max search depth of game tree
        self.max_width = max_width  # Max search width of game tree, not used in practice
        self.weight = np.asarray([[150, -80, 10, 10, -80, 150],  # weight matrix of board position
                                  [-80, -90, 5, 5, -90, -80],
                                  [10, 5, 1, 1, 5, 10],
                                  [10, 5, 1, 1, 5, 10],
                                  [-80, -90, 5, 5, -90, -80],
                                  [150, -80, 10, 10, -80, 150]])

    def play(self, board):
        _, result = self.alpha_beta(board, self.small_val, self.big_val, self.color, self.depth)
        return result

    def evaluate(self, board, color):
        return self.game.getScore(board, color)

    def alpha_beta(self, board, alpha, beta, color, depth):
        action = None
        max_val = self.small_val
        valids = self.game.getValidMoves(board, color)

        if np.sum(valids) is 0:
            return -self.alpha_beta(board, -beta, -alpha, -color, depth)[0], action
        if depth <= 0:
            return self.evaluate(board, color), action
        for a in range(self.game.getActionSize()):
            if valids[a]==0:
                continue
            nextBoard, _ = self.game.getNextState(board, color, a)
            val = -self.alpha_beta(nextBoard, -beta, -alpha, -color, depth - 1)[0]

            if val > max_val:
                max_val = val
                action = a
            if max_val > alpha:
                if max_val >= beta:
                    action = a
                    return max_val, action
                alpha = max_val
        return max_val, action

class SingleCorePlayer:

    def __init__(self, game, big_val=1e10, small_val=-1e10, max_depth=6, max_width=12):
        self.game = game
        self.color = 1  # Defining the color of the current player, use 'X' or 'O'
        self.big_val = big_val  # A value big enough for beta initialization
        self.small_val = small_val  # A value small enough for alpha initialization
        self.depth = max_depth  # Max search depth of game tree
        self.max_width = max_width  # Max search width of game tree, not used in practice
        self.weight = np.asarray([[150, -80, 10, 10, -80, 150],  # weight matrix of board position
                                  [-80, -90, 5, 5, -90, -80],
                                  [10, 5, 1, 1, 5, 10],
                                  [10, 5, 1, 1, 5, 10],
                                  [-80, -90, 5, 5, -90, -80],
                                  [150, -80, 10, 10, -80, 150]])
        self.factor = 50

    def play(self, board):
        global_depth = 36 - np.sum(board == 0)

        _, result = self.alpha_beta(board, self.small_val, self.big_val, self.color, self.depth,
                                    global_depth)
        return result

    def evaluate(self, board, color):
        _board = board * color
        sep_board = np.stack(((_board == 1).astype(int), np.negative((_board == -1).astype(int))))
        stability = 0
        for i in range(2):
            if sep_board[i, 0, 0]:
                stability += np.sum(sep_board[i, 0, 0:-1]) + np.sum(sep_board[i, 1::, 0])
                if sep_board[i, 1, 1]:
                    stability += np.sum(sep_board[i, 1, 1:-2]) + np.sum(sep_board[i, 2:-1, 1])
            if sep_board[i, 0, -1]:
                stability += np.sum(sep_board[i, 0:-1, -1]) + np.sum(sep_board[i, 0, 0:-1])
                if sep_board[i, 1, -2]:
                    stability += np.sum(sep_board[i, 1:-2, -2]) + np.sum(sep_board[i, 1, 1:-2])
            if sep_board[i, -1, -1]:
                stability += np.sum(sep_board[i, -1, 1::]) + np.sum(sep_board[i, 0:-1, -1])
                if sep_board[i, -2, -2]:
                    stability += np.sum(sep_board[i, -2, 2:-1]) + np.sum(sep_board[i, 1:-2, -2])
            if sep_board[i, -1, 0]:
                stability += np.sum(sep_board[i, 1::, 0]) + np.sum(sep_board[i, -1, 1::])
                if sep_board[i, -2, 1]:
                    stability += np.sum(sep_board[i, 2:-1, 1]) + np.sum(sep_board[i, -2, 2:-1])
        _board *= self.weight
        _board = np.sum(_board)
        _board += stability * self.factor
        return _board if np.sum(sep_board[0, :, :]) else self.small_val

    def alpha_beta(self, board, alpha, beta, color, depth, step):

        action = None
        max_val = self.small_val
        valids = self.game.getValidMoves(board, color)
        global_depth = step + self.depth - depth

        if np.sum(valids) is 0:
            return -self.alpha_beta(board, -beta, -alpha, -color, depth, step)[0], action

        if depth <= 0:
            mobility = np.sum(valids) * self.factor
            if global_depth < 16:
                return mobility, action
            return self.evaluate(board, color) + mobility, action

        for a in range(self.game.getActionSize()):
            if valids[a] == 0:
                continue
            nextBoard, _ = self.game.getNextState(board, color, a)
            val = -self.alpha_beta(nextBoard, -beta, -alpha, -color, depth - 1, step)[0]

            if val > max_val:
                max_val = val
                action = a
                if max_val > alpha:
                    if max_val >= beta:
                        return max_val, action
                    alpha = max_val

        return max_val, action
