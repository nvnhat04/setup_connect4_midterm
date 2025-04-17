import copy
from typing import List, Dict, Tuple
import numpy as np
import time

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

    def is_terminal(self) -> bool:
        return self.is_winner(1) or self.is_winner(2) or self.is_full()


class ConnectFourAI:
    def __init__(self):
        self.max_depth = 8  # Base depth
        self.CENTER_WEIGHT = 8  # Increased center preference
        self.THREE_WEIGHT = 100  # Increased for immediate threats
        self.TWO_WEIGHT = 5
        self.OPEN_THREE_WEIGHT = 150  # Special weight for open threes
        self.WIN_SCORE = 100000
        self.LOSE_SCORE = -100000
        self.DRAW_SCORE = 0
        self.time_limit = 5  # seconds for iterative deepening
        
    def order_moves(self, game: ConnectFour, moves: List[int], player: int) -> List[int]:
        """Improved move ordering with threat detection"""
        move_scores = []
        opponent = 3 - player
        
        for col in moves:
            score = 0
            
            # Center preference
            score += self.CENTER_WEIGHT * (3 - abs(3 - col))
            
            # Check for immediate wins
            game.make_move(col, player)
            if game.is_winner(player):
                game.undo_move(col)
                return [col]  # Return immediately if winning move found
            game.undo_move(col)
            
            # Check for opponent's immediate threats
            game.make_move(col, opponent)
            if game.is_winner(opponent):
                score += self.WIN_SCORE // 2  # Very high priority to block
            game.undo_move(col)
            
            # Evaluate potential to create threats
            score += self.evaluate_move_potential(game, col, player)
            
            move_scores.append((score, col))
        
        # Sort moves by score (descending)
        move_scores.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in move_scores]

    def evaluate_move_potential(self, game: ConnectFour, col: int, player: int) -> int:
        """Evaluate the potential of a move to create future threats"""
        if game.heights[col] >= game.rows:
            return -1000  # Invalid move
            
        row = game.heights[col]
        score = 0
        opponent = 3 - player
        
        # Check horizontal potential
        left = max(col - 3, 0)
        right = min(col + 3, game.cols - 1)
        window = [game.board[row][c] if game.heights[c] > row else -1 
                 for c in range(left, right + 1)]
        score += self._evaluate_window_potential(window, player, opponent)
        
        # Check vertical potential (only below)
        if row >= 3:
            window = [game.board[r][col] for r in range(row - 3, row + 1)]
            score += self._evaluate_window_potential(window, player, opponent)
        
        # Check diagonal (positive slope)
        for i in range(4):
            r = row - i
            c = col - i
            if r >= 0 and c >= 0 and r + 3 < game.rows and c + 3 < game.cols:
                window = [game.board[r+i][c+i] for i in range(4)]
                score += self._evaluate_window_potential(window, player, opponent)
        
        # Check diagonal (negative slope)
        for i in range(4):
            r = row + i
            c = col - i
            if r < game.rows and c >= 0 and r - 3 >= 0 and c + 3 < game.cols:
                window = [game.board[r-i][c+i] for i in range(4)]
                score += self._evaluate_window_potential(window, player, opponent)
        
        return score

    def _evaluate_window_potential(self, window: List[int], player: int, opponent: int) -> int:
        """Evaluate a window for potential threats"""
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)
        
        if opponent_count == 0:
            if player_count == 3 and empty_count == 1:
                return self.OPEN_THREE_WEIGHT  # Higher weight for open threes
            if player_count == 2 and empty_count == 2:
                return self.TWO_WEIGHT * 2  # Potential to become three
        elif player_count == 0:
            if opponent_count == 3 and empty_count == 1:
                return -self.THREE_WEIGHT * 2  # Need to block
        
        return 0

    def evaluate_position(self, game: ConnectFour, player: int, depth: int) -> int:
        """Enhanced evaluation function with pattern recognition"""
        opponent = 3 - player
        
        # Terminal state evaluations
        if game.is_winner(player):
            return self.WIN_SCORE + depth  # Prefer faster wins
        if game.is_winner(opponent):
            return self.LOSE_SCORE - depth  # Prefer slower losses
        if game.is_full():
            return self.DRAW_SCORE
        
        score = 0
        board = game.board
        
        # Evaluate all possible windows
        for r in range(game.rows):
            for c in range(game.cols - 3):
                window = [board[r][c+i] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)
                
        for c in range(game.cols):
            for r in range(game.rows - 3):
                window = [board[r+i][c] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)
                
        for r in range(game.rows - 3):
            for c in range(game.cols - 3):
                window = [board[r+i][c+i] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)
                
        for r in range(3, game.rows):
            for c in range(game.cols - 3):
                window = [board[r-i][c+i] for i in range(4)]
                score += self._evaluate_window(window, player, opponent)
        
        # Center control bonus
        center_cols = [2, 3, 4]
        for c in center_cols:
            if game.heights[c] > 0 and board[game.heights[c]-1][c] == player:
                score += self.CENTER_WEIGHT
        
        return score

    def _evaluate_window(self, window: List[int], player: int, opponent: int) -> int:
        """Evaluate a single window with more sophisticated patterns"""
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)
        
        if opponent_count == 0:
            if player_count == 4:
                return self.WIN_SCORE
            if player_count == 3 and empty_count == 1:
                return self.THREE_WEIGHT * 2  # Stronger weight for open threes
            if player_count == 2 and empty_count == 2:
                return self.TWO_WEIGHT
            if player_count == 1 and empty_count == 3:
                return self.TWO_WEIGHT // 2
            
        elif player_count == 0:
            if opponent_count == 4:
                return self.LOSE_SCORE
            if opponent_count == 3 and empty_count == 1:
                return -self.THREE_WEIGHT * 3  # Must block opponent's open three
            if opponent_count == 2 and empty_count == 2:
                return -self.TWO_WEIGHT
        
        return 0

    def negamax(self, game: ConnectFour, depth: int, alpha: int, beta: int, 
                player: int, start_time: float, max_time: float) -> Tuple[int, int]:
        """Negamax with alpha-beta pruning and time management"""
        # Check time limit
        if time.time() - start_time > max_time:
            return 0, -1  # Timeout
            
        hash_key = game.get_board_hash()
        if hash_key in game.transposition_table:
            tt_value, tt_depth, tt_flag = game.transposition_table[hash_key]
            if tt_depth >= depth:
                if tt_flag == 0:  # Exact value
                    return tt_value, -1
                elif tt_flag == 1:  # Lower bound
                    alpha = max(alpha, tt_value)
                elif tt_flag == 2:  # Upper bound
                    beta = min(beta, tt_value)
                if alpha >= beta:
                    return tt_value, -1

        if depth == 0 or game.is_terminal():
            return self.evaluate_position(game, player, depth), -1

        valid_moves = self.order_moves(game, game.get_valid_moves(), player)
        if not valid_moves:
            return self.evaluate_position(game, player, depth), -1

        best_move = valid_moves[0]
        best_value = float('-inf')

        for move in valid_moves:
            game.make_move(move, player)
            value = -self.negamax(game, depth - 1, -beta, -alpha, 
                                 3 - player, start_time, max_time)[0]
            game.undo_move(move)

            if value > best_value:
                best_value = value
                best_move = move
                if best_value >= beta:
                    break
            alpha = max(alpha, best_value)
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
        """Iterative deepening with time management"""
        start_time = time.time()
        best_move = 3  # Default center move
        max_depth = self.max_depth
        
        # Count pieces to determine game phase
        pieces = sum(sum(1 for cell in row if cell != 0) for row in game.board)
        
        # Adjust time and depth based on game phase
        if pieces < 8:  # Opening
            max_depth = min(8, self.max_depth + 2)
        elif pieces < 24:  # Midgame
            max_depth = min(10, self.max_depth + 4)
        else:  # Endgame
            max_depth = min(12, self.max_depth + 6)
        
        # Iterative deepening
        for depth in range(1, max_depth + 1):
            try:
                score, move = self.negamax(
                    game, depth, float('-inf'), float('inf'), 
                    player, start_time, self.time_limit
                )
                if move != -1:  # Only update if we got a valid move
                    best_move = move
            except TimeoutError:
                break  # Stop searching if we're out of time
                
        print(f"Chose move {best_move} at depth {depth-1} with score {score}")
        return best_move


def main():
    # Test case from original code
    current_player = 1
    board_state =[  [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 2, 0, 0, 0],
                    [1, 2, 2, 2, 1, 0, 0]]

    game = ConnectFour()
    ai = ConnectFourAI()

    # Load the board state
    reversed_board = [row[:] for row in reversed(board_state)]
    game.board = copy.deepcopy(reversed_board)
    
    # Calculate heights
    for col in range(game.cols):
        for row in range(game.rows):
            if game.board[row][col] != 0:
                game.heights[col] = row + 1
    
    # Determine current player
    player1_pieces = sum(row.count(1) for row in game.board)
    player2_pieces = sum(row.count(2) for row in game.board)
    current_player = 1 if player1_pieces == player2_pieces else 2
    
    print("Current board state:")
    for row in reversed(game.board):
        print(row)
    print(f"\nCurrent player: {current_player}")
    print(f"Valid moves: {game.get_valid_moves()}")
    
    # Get AI's move
    best_move = ai.find_best_move(game, current_player)
    print(f"\nAI (Player {current_player}) chooses column: {best_move}")
    
    # Make the move
    if game.make_move(best_move, current_player):
        print("\nNew board state:")
        for row in reversed(game.board):
            print(row)
        
        if game.is_winner(current_player):
            print(f"\nPlayer {current_player} wins!")
        elif game.is_full():
            print("\nThe game is a draw!")
    else:
        print("Invalid move selected by AI")

if __name__ == "__main__":
    main()