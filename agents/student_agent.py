# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
import time
from helpers import random_move, execute_move, check_endgame, get_valid_moves, MoveCoordinates

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    Ataxx Agent using Alpha-Beta Pruning with iterative deepening
    and a comprehensive evaluation function.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.time_limit = 1.9  
        
        self.board_size = 7
        self.corners = [(0, 0), (0, 6), (6, 0), (6, 6)]
        
        # Positional weight matrix for board evaluation
        # Corners are highly valuable, edges are risky, center is moderately good
        self.position_weights = np.array([
            [10, -5,  2,  2,  2, -5, 10],
            [-5, -5,  1,  1,  1, -5, -5],
            [ 2,  1,  3,  3,  3,  1,  2],
            [ 2,  1,  3,  3,  3,  1,  2],
            [ 2,  1,  3,  3,  3,  1,  2],
            [-5, -5,  1,  1,  1, -5, -5],
            [10, -5,  2,  2,  2, -5, 10]
        ])
        
        # All 8 possible directions for adjacent squares
        self.directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), 
                          (0, 1), (1, -1), (1, 0), (1, 1)]

    def step(self, chess_board, player, opponent):
        """
        Main method called by game engine to get the next move.
        Uses iterative deepening with alpha-beta pruning.
        """
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
        
        # Iterative deepening: start with depth 1, increase until time runs out
        max_depth = 3  
        depth = 1
        
        while depth <= max_depth and time.time() - start_time < self.time_limit * 0.8:
            current_best = None
            current_score = -float('inf')
            alpha = -float('inf')
            beta = float('inf')
            
            # Order moves for better pruning efficiency
            ordered_moves = self.prioritize_moves(chess_board, player, valid_moves)
            
            for move in ordered_moves:
                if time.time() - start_time > self.time_limit:
                    break

                board_copy = chess_board.copy()
                execute_move(board_copy, move, player)
                
                # Evaluate the move using alpha-beta search
                score = self.alpha_beta_search(
                    board_copy, opponent, player, 
                    depth - 1, alpha, beta, False, start_time
                )
                
                if score > current_score:
                    current_score = score
                    current_best = move
                
                alpha = max(alpha, current_score)
            
            # Only update best move if we completed this depth level
            if current_best and time.time() - start_time < self.time_limit:
                best_move = current_best
                best_score = current_score
            
            depth += 1
        
        time_taken = time.time() - start_time
        print("My AI's turn took", time_taken, "seconds")
        
        return best_move
    
    def alpha_beta_search(self, board, current_player, original_player, 
                          depth, alpha, beta, maximizing, start_time):
        """
        Recursive alpha-beta pruning search.
        Returns the heuristic value of the board position.
        """
        if time.time() - start_time > self.time_limit:
            return 0
        
        # Base case: reached depth limit or terminal state
        if depth == 0:
            return self.evaluate_position(board, original_player)
        
        # Get moves for current player
        moves = get_valid_moves(board, current_player)
        
        # If no moves available, check if game is over
        if not moves:
            is_endgame, p0_score, p1_score = check_endgame(board)
            if is_endgame:
                return self.terminal_evaluation(board, original_player, p0_score, p1_score)
            # Pass turn to opponent
            opponent = 1 if current_player == 2 else 2
            return self.alpha_beta_search(
                board, opponent, original_player, 
                depth - 1, alpha, beta, not maximizing, start_time
            )
        
        # Order moves for better pruning
        moves = self.prioritize_moves(board, current_player, moves)
        
        if maximizing:
            max_eval = -float('inf')
            for move in moves:
                if time.time() - start_time > self.time_limit:
                    break

                board_copy = board.copy()
                execute_move(board_copy, move, current_player)
                
                # Recursive evaluation
                eval_score = self.alpha_beta_search(
                    board_copy, 
                    1 if current_player == 2 else 2,  # Switch player
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
                
                eval_score = self.alpha_beta_search(
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
    
    def evaluate_position(self, board, player):
        """
        Comprehensive board evaluation function.
        Considers material, position, mobility, and corner control.
        """
        opponent = 1 if player == 2 else 2
        
        # 1. Piece count advantage 
        player_pieces = np.sum(board == player)
        opponent_pieces = np.sum(board == opponent)
        material_score = player_pieces - opponent_pieces
        
        # 2. Position advantage 
        player_position = np.sum(self.position_weights[board == player])
        opponent_position = np.sum(self.position_weights[board == opponent])
        positional_score = player_position - opponent_position
        
        # 3. Mobility advantage
        mobility = self.approx_mobility(board, player) - self.approx_mobility(board, opponent)
        
        # 4. Corner control
        player_corners = sum(1 for corner in self.corners if board[corner] == player)
        opponent_corners = sum(1 for corner in self.corners if board[corner] == opponent)
        corner_score = player_corners - opponent_corners
        
        # Weighted combination of all factors
        total_score = (
            material_score * 50 +           # Most important: piece count
            corner_score * 40 +             # Very important: corner control
            positional_score * 5 +          # Moderately important: board position
            mobility * 2                    # Less important: mobility
        )
        
        return total_score
    
    def approx_mobility(self, board, player):
        """
        Approximation of mobility by counting empty squares adjacent to player's pieces.
        This is faster than generating all valid moves.
        """
        count = 0
        for r in range(self.board_size):
            for c in range(self.board_size):
                if board[r, c] == player:
                    # Check all 8 directions for empty adjacent squares
                    for dr, dc in self.directions:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                            if board[nr, nc] == 0: 
                                count += 1
        return count
    
    def prioritize_moves(self, board, player, moves):
        """
        Order moves for better alpha-beta pruning.
        Good moves first -> more pruning -> deeper search.
        """
        if len(moves) <= 1:
            return moves
        
        scored_moves = []
        opponent = 1 if player == 2 else 2
        
        for move in moves:
            score = 0
            dest = move.get_dest()
            
            # 1. Corner moves are best
            if dest in self.corners:
                score += 100
            
            # 2. Moves that capture many pieces are good
            captures = 0
            for dr, dc in self.directions:
                nr, nc = dest[0] + dr, dest[1] + dc
                if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                    if board[nr, nc] == opponent:
                        captures += 1
            score += captures * 20
            
            # 3. Positional value from weight matrix
            score += self.position_weights[dest]
            
            scored_moves.append((score, move))
        
        # Sort by score (highest first) and return top moves
        scored_moves.sort(reverse=True, key=lambda x: x[0])
        # Limit to top 15 moves to save time
        return [move for _, move in scored_moves[:min(len(scored_moves), 15)]]
    
    def terminal_evaluation(self, board, player, p0_score, p1_score):
        """
        Evaluate terminal game state (game over).
        Returns very high positive for win, very low negative for loss.
        """
        if player == 1:
            my_score = p0_score
            opp_score = p1_score
        else:
            my_score = p1_score
            opp_score = p0_score
        
        if my_score > opp_score:
            return 10000  # Win
        elif my_score < opp_score:
            return -10000  # Loss
        else:
            return 0  # Draw
