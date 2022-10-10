import random
import copy

RED_PLAYER_VAL = -1
YELLOW_PLAYER_VAL = 1


class Player:

    def __init__(self, value, strategy='random', model=None):
        self.value = value
        self.strategy = strategy
        self.model = model

    def getMove(self, availableMoves, board):
        if self.strategy == "random":
            return availableMoves[random.randrange(0, len(availableMoves))]
        elif self.strategy == "human":
            while True:
                move = input("Enter a move(0-6, q to exit): ")
                if move == 'q':
                    return move
                if move.isdigit():
                    move = int(move)
                    for av_move in availableMoves:
                        if av_move[1] == move:
                            return av_move
                print("Invalid move, try again.")
        else:
            maxValue = 0
            bestMove = availableMoves[0]
            for availableMove in availableMoves:
                boardCopy = copy.deepcopy(board)
                boardCopy[availableMove[0]][availableMove[1]] = self.value
                if self.value == RED_PLAYER_VAL:
                    value = self.model.predict(boardCopy, 2)
                else:
                    value = self.model.predict(boardCopy, 0)
                if value > maxValue:
                    maxValue = value
                    bestMove = availableMove
            return bestMove

    def getPlayer(self):
        return self.value
