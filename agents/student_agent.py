# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves, MoveCoordinates

@register_agent("random_agent")
class RandomAgent(Agent):

    def __init__(self):
        super(RandomAgent, self).__init__()
        self.name = "RandomAgent"
        self.root_player = None
        self.time_limit = 1.9 
        
        self.board_size = 7
        self.corners = [(0, 0), (0, 6), (6, 0), (6, 6)]
        
        self.position_weights = np.array([
            [10, -5,  1,  1,  1, -5, 10],
            [-5, -5,  2,  2,  2, -5, -5],
            [ 1,  2,  3,  3,  3,  2,  1],
            [ 1,  2,  3,  5,  3,  2,  1],
            [ 1,  2,  3,  3,  3,  2,  1],
            [-5, -5,  2,  2,  2, -5, -5],
            [10, -5,  1,  1,  1, -5, 10]
        ])
        
        self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


    def step(self, chess_board, player, opponent):
        self.root_player = player

        valid_moves = get_valid_moves(chess_board, player)
        if not valid_moves:
            return None
        
        immediate_best_action = self.get_immediate_best_move(chess_board, valid_moves, player)
        if immediate_best_action is not None:
            return immediate_best_action
        
        depth = 1
        max_depth = 3 

        best_overall_move = valid_moves[np.random.randint(len(valid_moves))]

        end_time = time.time() + self.time_limit
        while depth <= max_depth and time.time() < end_time:
            alpha = -float('inf')
            beta = float('inf')

            current_best = None
            current_score = -float('inf')
            
            completed_depth = True
            
            ordered_moves = self.fast_order_moves(chess_board, player, valid_moves)
            for move in ordered_moves:
                if time.time() >= end_time:
                    completed_depth = False
                    break    

                board_copy = chess_board.copy()
                execute_move(board_copy, move, player)
                
                score = self.alpha_beta_minimax(
                    board_copy, opponent, 
                    depth - 1, alpha, beta, False
                )
                
                if score > current_score:
                    current_score = score
                    current_best = move
                
                alpha = max(alpha, current_score)
            
            if completed_depth:
                best_overall_move = current_best
                depth += 1
        
        return best_overall_move
    

    def alpha_beta_minimax(self, board, player, depth, alpha, beta, maximizing):
        if depth == 0:
            return self.fast_evaluate(board)
        
        opponent = 3 - player
        moves = get_valid_moves(board, player)
        if not moves:
            is_endgame, p1_score, p2_score = check_endgame(board)
            if is_endgame:
                return self.terminal_evaluation(p1_score, p2_score)
        
            return self.alpha_beta_minimax(
                board, opponent, depth - 1, alpha, beta, not maximizing
            )
        
        moves = self.fast_order_moves(board, player, moves)
        if maximizing:
            max_eval = -float('inf')
            for move in moves:
                board_copy = board.copy()
                execute_move(board_copy, move, player)
                
                eval_score = self.alpha_beta_minimax(
                    board_copy, 
                    opponent,  
                    depth - 1, alpha, beta, False
                )
                
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  
            
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                board_copy = board.copy()
                execute_move(board_copy, move, player)
                
                eval_score = self.alpha_beta_minimax(
                    board_copy,
                    opponent,

                    depth - 1, alpha, beta, True
                )
                
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  
            
            return min_eval
    

    def fast_evaluate(self, board):
        player = self.root_player
        opponent = 3 - player
        
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
    

    def terminal_evaluation(self, p1_score, p2_score):
        root_player = self.root_player

        if p1_score > p2_score:
            return 10000 if root_player == 1 else -10000
        elif p2_score > p1_score:
            return 10000 if root_player == 2 else -10000
        else:
            return 0
    

    def get_immediate_best_move(self, board, moves, player):
        if len(moves) == 1:
            return moves[0]
        
        root_player = player
        corners = self.corners
        for m in moves:
            tmp = board.copy()

            execute_move(tmp, m, root_player)
            is_end, s1, s2 = check_endgame(tmp)
            move_dest = m.get_dest()

            if is_end:
                if (root_player == 1 and s1 > s2) or (root_player == 2 and s2 > s1):
                    return m
            elif move_dest in corners:
                return m
        
        return None