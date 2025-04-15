import copy
from typing import List, Dict, Tuple
import numpy as np
# from bot_2 import minimax
class ConnectFour:
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.board = [[0] * self.cols for _ in range(self.rows)]
        self.heights = [0] * self.cols  # Tracks height of each column
        self.transposition_table: Dict[int, Tuple[int, int, int]] = {}
        
    def make_move(self, col: int, player: int) -> bool:
        if col < 0 or col >= self.cols or self.heights[col] >= self.rows:
            return False
        row = self.heights[col]
        self.board[row][col] = player
        self.heights[col] += 1
        return True

    def undo_move(self, col: int) -> None:
        if self.heights[col] > 0:
            self.heights[col] -= 1
            self.board[self.heights[col]][col] = 0

    def is_winner(self, player: int) -> bool:
        # Check horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if all(self.board[r][c+i] == player for i in range(4)):
                    return True
                    
        # Check vertical
        for c in range(self.cols):
            for r in range(self.rows - 3):
                if all(self.board[r+i][c] == player for i in range(4)):
                    return True
                    
        # Check diagonal (positive slope)
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if all(self.board[r+i][c+i] == player for i in range(4)):
                    return True
                    
        # Check diagonal (negative slope)
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if all(self.board[r-i][c+i] == player for i in range(4)):
                    return True
        return False

    def is_full(self) -> bool:
        return all(h == self.rows for h in self.heights)

    def get_valid_moves(self) -> List[int]:
        return [col for col in range(self.cols) if self.heights[col] < self.rows]

    def get_board_hash(self) -> int:
        return hash(str(self.board))

class ConnectFourAI:
    # def __init__(self):
    #     self.max_depth = 8
    #     self.CENTER_WEIGHT = 4
    #     self.THREE_WEIGHT = 12
    #     self.TWO_WEIGHT = 5
    def __init__(self):
        self.max_depth = 6
        self.CENTER_WEIGHT = 6
        self.THREE_WEIGHT = 50
        self.TWO_WEIGHT = 2
        self.WIN_SCORE = 1000       # Score for actual wins
        self.LOSE_SCORE = -1000     # Score for losses
        self.DRAW_SCORE = 0         # Score for draws

    def order_moves(self, game: ConnectFour, moves: List[int], player: int) -> List[int]:
        # Prioritize center columns and potentially winning moves
        scores = []
        for col in moves:
            score = self.CENTER_WEIGHT * (3 - abs(3 - col))  # Center preference
            game.make_move(col, player)
            if game.is_winner(player):
                game.undo_move(col)
                return [col]  # Immediate win
            game.undo_move(col)
            scores.append((score, col))
        return [move for _, move in sorted(scores, reverse=True)]

    def evaluate_position(self, game: ConnectFour, player: int, depth: int) -> int:
        if game.is_winner(player):
            return self.WIN_SCORE - depth  # Prefer faster wins
        if game.is_winner(3 - player):  # Opponent wins
            return self.LOSE_SCORE + depth  # Prefer slower losses
        if game.is_full():
            return self.DRAW_SCORE
        
        # Heuristic evaluation for non-terminal positions
        score = 0
        board = game.board
        
        # Check all possible four-in-a-row positions
        for r in range(game.rows):
            for c in range(game.cols - 3):
                window = [board[r][c+i] for i in range(4)]
                score += self._evaluate_window(window, player)
                
        for c in range(game.cols):
            for r in range(game.rows - 3):
                window = [board[r+i][c] for i in range(4)]
                score += self._evaluate_window(window, player)
                
        for r in range(game.rows - 3):
            for c in range(game.cols - 3):
                window = [board[r+i][c+i] for i in range(4)]
                score += self._evaluate_window(window, player)
                
        for r in range(3, game.rows):
            for c in range(game.cols - 3):
                window = [board[r-i][c+i] for i in range(4)]
                score += self._evaluate_window(window, player)
                
        return score

    def _evaluate_window(self, window: List[int], player: int) -> int:
        opponent = 3 - player
        player_count = window.count(player)
        empty_count = window.count(0)
        
        if player_count == 3 and empty_count == 1:
            return self.THREE_WEIGHT
        if player_count == 2 and empty_count == 2:
            return self.TWO_WEIGHT
        if window.count(opponent) == 3 and empty_count == 1:
            return -self.THREE_WEIGHT
        if window.count(opponent) == 2 and empty_count == 2:
            return -self.TWO_WEIGHT
        return 0

    def negamax(self, game: ConnectFour, depth: int, alpha: int, beta: int, player: int) -> Tuple[int, int]:
        hash_key = game.get_board_hash()
        if hash_key in game.transposition_table:
            tt_value, tt_depth, tt_flag = game.transposition_table[hash_key]
            if tt_depth >= depth:
                if tt_flag == 0:  # Exact value nvnhat04
                    return tt_value, -1
                elif tt_flag == 1:  # Lower bound
                    alpha = max(alpha, tt_value)
                elif tt_flag == 2:  # Upper bound
                    beta = min(beta, tt_value)
                if alpha >= beta:
                    return tt_value, -1

        if depth == 0 or game.is_winner(1) or game.is_winner(2) or game.is_full():
            return self.evaluate_position(game, player, depth), -1

        valid_moves = self.order_moves(game, game.get_valid_moves(), player)
        # print(f"Valid moves: {valid_moves}")
        if not valid_moves:
            return self.evaluate_position(game, player, depth), -1

        best_move = valid_moves[0]
        best_value = float('-inf')

        for move in valid_moves:
            game.make_move(move, player)
            value = -self.negamax(game, depth - 1, -beta, -alpha, 3 - player)[0]
            game.undo_move(move)

            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        # Store in transposition table
        if best_value <= alpha:
            flag = 2  # Upper bound
        elif best_value >= beta:
            flag = 1  # Lower bound
        else:
            flag = 0  # Exact value
        game.transposition_table[hash_key] = (best_value, depth, flag)

        return best_value, best_move

    def find_best_move(self, game: ConnectFour, player: int) -> int:
        # print("player:", player)
        cnt_player = 0
        for row in game.board:
            cnt_player += row.count(player)
        # if(cnt_player < 3 ):#nvnhat
        #     print("game.board[5]:", game.board[0])
        #     print(game.board[0].count(3-player))
        #     if(game.board[0].count(3-player) > 1):
        #         return 4 if game.get_valid_moves().count(4) > 0 else 2
        #     return 3 if game.get_valid_moves().count(3) > 0 else 4 
        # depth = self.max_depth
        if(cnt_player < 5):
            depth = self.max_depth
        elif(cnt_player < 10):
            depth = self.max_depth + 2
        elif(cnt_player < 15):
            depth = self.max_depth + 3
        else:
            depth = self.max_depth + 4
        score, move = self.negamax(game, depth, float('-inf'), float('inf'), player)

        print(f"Best move: {move}, Score: {score}")
        return move


def main():
    # Input format
    current_player = 1
    valid_moves = [0, 1, 2, 3, 4, 5, 6]
    board_state =[  [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 2, 0, 0, 0],
                    [1, 2, 2, 2, 1, 0, 0]]

    # Initialize the game and AI
    game = ConnectFour()
    ai = ConnectFourAI()

    # Load the board state into the game
    reversed_board = [row[:] for row in reversed(board_state)]
    
    # Load the board state into the game
    game.board = copy.deepcopy(reversed_board)
    
    # Calculate the heights based on the board state
    for col in range(game.cols):
        for row in range(game.rows):
            if game.board[row][col] != 0:
                game.heights[col] = row + 1
    
    # Determine whose turn it is based on the number of pieces
    player1_pieces = sum(row.count(1) for row in game.board)
    player2_pieces = sum(row.count(2) for row in game.board)
    current_player = 1 if player1_pieces == player2_pieces else 2
    
    print("Current board state:")
    for row in reversed(game.board):
        print(row)
    print(f"\nCurrent player: {current_player}")
    print(f"Valid moves: {game.get_valid_moves()}")
    
    # Get AI's move
    ai_player = current_player
    best_move = ai.find_best_move(game, ai_player)
    print(f"\nAI (Player {ai_player}) chooses column: {best_move}")
    
    # Make the move
    if game.make_move(best_move, ai_player):
        print("\nNew board state:")
        for row in reversed(game.board):
            print(row)
        
        # Check for winner
        if game.is_winner(ai_player):
            print(f"\nPlayer {ai_player} wins!")
        elif game.is_full():
            print("\nThe game is a draw!")
    else:
        print("Invalid move selected by AI")

if __name__ == "__main__":
    main()
export = {ConnectFour, ConnectFourAI}    

# Current player: 1
# Valid moves: [0, 2, 3, 4, 5, 6]
# Board state: [
#  [0, 1, 0, 0, 0, 0, 0],
#  [0, 2, 0, 0, 0, 0, 0],
#  [0, 1, 0, 1, 2, 0, 0],
#  [2, 2, 0, 1, 2, 0, 2],
#  [2, 1, 2, 1, 2, 1, 1],
#  [1, 2, 1, 2, 1, 2, 1]]
#  nvnhat04