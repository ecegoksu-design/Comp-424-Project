# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves, MoveCoordinates

@register_agent("student_agent")
class StudentAgent(Agent):

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.time_limit = 1.8  
        
        self.board_size = 7
        self.corners = [(0, 0), (0, 6), (6, 0), (6, 6)]
        
        self.position_weights = np.array([
            [10, -5,  2,  2,  2, -5, 10],
            [-5, -5,  1,  1,  1, -5, -5],
            [ 2,  1,  3,  3,  3,  1,  2],
            [ 2,  1,  3,  3,  3,  1,  2],
            [ 2,  1,  3,  3,  3,  1,  2],
            [-5, -5,  1,  1,  1, -5, -5],
            [10, -5,  2,  2,  2, -5, 10]
        ])
        
        self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def step(self, chess_board, player, opponent):
        start_time = time.time()
        
        valid_moves = get_valid_moves(chess_board, player)
        if not valid_moves:
            return None
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        for move in valid_moves:
            dest = move.get_dest()
            if dest in self.corners:
                return move
        
        best_move = valid_moves[0]
        best_score = -float('inf')
        
        max_depth = 3  
        depth = 1
        
        while depth <= max_depth and time.time() - start_time < self.time_limit * 0.8:
            current_best = None
            current_score = -float('inf')
            alpha = -float('inf')
            beta = float('inf')
            
            ordered_moves = self.fast_order_moves(chess_board, player, valid_moves)
            
            for move in ordered_moves:
                if time.time() - start_time > self.time_limit:
                    break
                    
                board_copy = chess_board.copy()
                execute_move(board_copy, move, player)
                
                score = self.alpha_beta_minimax(
                    board_copy, opponent, player, 
                    depth - 1, alpha, beta, False, start_time
                )
                
                if score > current_score:
                    current_score = score
                    current_best = move
                
                alpha = max(alpha, current_score)
            
            if current_best and time.time() - start_time < self.time_limit:
                best_move = current_best
                best_score = current_score
            
            depth += 1
        
        time_taken = time.time() - start_time
        if time_taken > 0.1:  
            print(f"StudentAgent: {time_taken:.2f}s, depth {depth-1}")
        
        return best_move
    
    def alpha_beta_minimax(self, board, current_player, original_player, 
                          depth, alpha, beta, maximizing, start_time):
        if time.time() - start_time > self.time_limit:
            return 0
        
        if depth == 0:
            return self.fast_evaluate(board, original_player)
        
        moves = get_valid_moves(board, current_player)
        
        if not moves:
            is_endgame, p0_score, p1_score = check_endgame(board)
            if is_endgame:
                return self.terminal_evaluation(board, original_player, p0_score, p1_score)
            opponent = 1 if current_player == 2 else 2
            return self.alpha_beta_minimax(
                board, opponent, original_player, 
                depth - 1, alpha, beta, not maximizing, start_time
            )
        
        moves = self.fast_order_moves(board, current_player, moves)
        
        if maximizing:
            max_eval = -float('inf')
            for move in moves:
                if time.time() - start_time > self.time_limit:
                    break
                
                board_copy = board.copy()
                execute_move(board_copy, move, current_player)
                
                eval_score = self.alpha_beta_minimax(
                    board_copy, 
                    1 if current_player == 2 else 2,  
                    original_player,
                    depth - 1, alpha, beta, False, start_time
                )
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  
            
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                if time.time() - start_time > self.time_limit:
                    break
                
                board_copy = board.copy()
                execute_move(board_copy, move, current_player)
                
                eval_score = self.alpha_beta_minimax(
                    board_copy,
                    1 if current_player == 2 else 2,
                    original_player,
                    depth - 1, alpha, beta, True, start_time
                )
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  
            
            return min_eval
    
    def fast_evaluate(self, board, player):
        opponent = 1 if player == 2 else 2
        
        player_pieces = np.sum(board == player)
        opponent_pieces = np.sum(board == opponent)
        material = player_pieces - opponent_pieces
        
        player_pos = np.sum(self.position_weights[board == player])
        opponent_pos = np.sum(self.position_weights[board == opponent])
        positional = player_pos - opponent_pos
        
        mobility = self.approximate_mobility(board, player) - self.approximate_mobility(board, opponent)
        
        player_corners = sum(1 for corner in self.corners if board[corner] == player)
        opponent_corners = sum(1 for corner in self.corners if board[corner] == opponent)
        corners = player_corners - opponent_corners
        
        score = (
            material * 50 +           
            corners * 40 +            
            positional * 5 +          
            mobility * 2              
        )
        
        return score
    
    def approximate_mobility(self, board, player):
        count = 0
        for r in range(7):
            for c in range(7):
                if board[r, c] == player:
                    for dr, dc in self.directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < 7 and 0 <= nc < 7 and board[nr, nc] == 0:
                            count += 1
        return count
    
    def fast_order_moves(self, board, player, moves):
        if len(moves) <= 1:
            return moves
        
        scored_moves = []
        for move in moves:
            score = 0
            dest = move.get_dest()
            
            if dest in self.corners:
                score += 100
            
            captures = 0
            for dr, dc in self.directions:
                nr, nc = dest[0] + dr, dest[1] + dc
                if 0 <= nr < 7 and 0 <= nc < 7:
                    if board[nr, nc] == (1 if player == 2 else 2):
                        captures += 1
            score += captures * 20
            
            score += self.position_weights[dest]
            
            scored_moves.append((score, move))
        
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in scored_moves[:min(len(scored_moves), 15)]]  
    
    def terminal_evaluation(self, board, player, p0_score, p1_score):
        if player == 1:
            my_score = p0_score
            opp_score = p1_score
        else:
            my_score = p1_score
            opp_score = p0_score
        
        if my_score > opp_score:
            return 10000
        elif my_score < opp_score:
            return -10000
        else:
            return 0