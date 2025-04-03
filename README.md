# Connect 4 AI API

Ứng dụng API cho phép tích hợp thuật toán AI vào hệ thống Connect 4.

## API Endpoint

Sau khi triển khai API của bạn, bạn sẽ cần cung cấp URL endpoint cho server chính:
```
https://your-ai-service.com/api/connect4-move
```

### Ví dụ URL sau khi triển khai bằng Ngrok:
```
https://c3b1-2405-4802-21ad-48b0-7c4a-8729-ca-4c80.ngrok-free.app/api/connect4-move
```

## Format API

### Request Format
```json
{
  "board": [[0,0,0,...], [...], ...],
  "current_player": 1,
  "valid_moves": [0,1,2,...]
}
```

Trong đó:
- `board`: Mảng 2 chiều (6x7) biểu diễn bảng Connect 4
  - `0`: Ô trống
  - `1`: Quân của người chơi 1
  - `2`: Quân của người chơi 2
- `current_player`: Người chơi hiện tại (1 hoặc 2)
- `valid_moves`: Các cột còn có thể đặt quân (từ 0-6)

### Response Format
```json
{
  "move": 3
}
```

Trong đó:
- `move`: Cột mà AI quyết định đặt quân (chỉ số từ 0-6)

Ví dụ: Nếu API trả về `move = 3`, server sẽ đặt quân vào ô trống thấp nhất của cột thứ 4 (vì chỉ số bắt đầu từ 0).

## Triển khai API

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```

### File app.py
```python
from fastapi import FastAPI, HTTPException
import random
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware

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
        if not game_state.valid_moves:
            raise ValueError("Không có nước đi hợp lệ")
            
        # TODO: Thay thế mã này bằng thuật toán AI của bạn
        selected_move = random.choice(game_state.valid_moves)
        
        return AIResponse(move=selected_move)
    except Exception as e:
        if game_state.valid_moves:
            return AIResponse(move=game_state.valid_moves[0])
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

### Chạy server
```bash
python app.py
```

## Triển khai public với Ngrok

Để server của bạn có thể truy cập được từ internet, bạn có thể sử dụng Ngrok:

1. Tải và cài đặt Ngrok: https://ngrok.com/download
2. Chạy server FastAPI của bạn (mặc định cổng 8080)
3. Trong terminal khác, chạy lệnh:
```bash
ngrok http 8080
```
4. Sao chép URL Forwarding (dạng https://xxxx-xxxx.ngrok-free.app) và đăng ký với server chính.

## Phát triển thuật toán AI

Để cải thiện AI của bạn, hãy thay thế đoạn mã sau trong hàm `make_move`:

```python
# TODO: Thay thế mã này bằng thuật toán AI của bạn
selected_move = random.choice(game_state.valid_moves)
```

Bạn có thể cài đặt các thuật toán như Minimax, Alpha-Beta Pruning, hoặc các kỹ thuật Machine Learning để cải thiện khả năng chơi của AI.

## Ví dụ Game State

```json
{
  "board": [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
  ],
  "current_player": 1,
  "valid_moves": [0, 1, 2, 3, 4, 5, 6]
}
```

## Lưu ý

- API của bạn sẽ tự động nhận dữ liệu từ server và chuyển đổi thành đối tượng `GameState`
- Bạn chỉ cần tập trung vào việc phát triển thuật toán AI để chọn nước đi tốt nhất
- Đảm bảo API của bạn luôn trả về một nước đi hợp lệ (nằm trong danh sách `valid_moves`)
- Nếu xảy ra lỗi, API sẽ tự động chọn nước đi đầu tiên trong danh sách `valid_moves`
``` 