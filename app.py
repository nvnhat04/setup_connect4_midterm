from fastapi import FastAPI, HTTPException
import random
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from bot import ConnectFourAI, ConnectFour
import copy
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameState(BaseModel):
    board: List[List[int]]
    current_player: int
    valid_moves: List[int]

class AIResponse(BaseModel):
    move: int

@app.post("/api/connect4-move")
async def make_move(game_state: GameState) -> AIResponse:
    try:
        # print("Received game state:")
        # # print(game_state)
        # print(f"Current player: {game_state.current_player}")
        # print(f"Valid moves: {game_state.valid_moves}")
        # print(f"Board state:{game_state.board}")
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")

        # Initialize the game and AI
        game = ConnectFour()
        ai = ConnectFourAI()

        # Load the board state into the game
        reversed_board = [row[:] for row in reversed(game_state.board)]
        
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
     
        # selected_move =  random.choice(game_state.valid_moves) # change logic thuật toán AI của bạn ở đây
        # print(f"AI selected move: {selected_move}")
        return AIResponse(move=best_move)
    except Exception as e:
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)