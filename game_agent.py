"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random

class FoundWinningMoveException(Exception):
    def __init__(self, move):
        self.move = move


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def moves_intersect(game): #returns True if the following possible moves of players overlap
    def mil(move, game):
        idx = move[0] + move[1] * game.height
        return 0 <= move[0] < game.height and 0 <= move[1] < game.width and game._board_state[idx] == 0

    p1loc = game.get_player_location(game._player_1)
    p2loc = game.get_player_location(game._player_2)
    directions = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                  (1, -2), (1, 2), (2, -1), (2, 1)]
    if p1loc is None:
        p1moves = game.get_blank_spaces()
    else:
        r,c = p1loc
        p1moves = [(r + dr, c + dc) for dr, dc in directions
                       if mil((r + dr, c + dc), game)]

    if p2loc is None:
        p2moves = game.get_blank_spaces()
    else:
        r, c = p2loc
        p2moves = [(r + dr, c + dc) for dr, dc in directions
                   if mil((r + dr, c + dc), game)]
    return bool(set(p1moves) & set(p2moves))/2


def possible_moves_count(row, column, game): # returns the number of possible moves if the board was empty
    total_moves = (8
                   - (1 if (column < 2 or row < 1) else 0)  # 10 o'clock blocked
                   - (1 if (column < 1 or row < 2) else 0)  # 11 o'clock blocked
                   - (1 if (column > game.width - 2 or row < 2) else 0)  # 1 o'clock blocked
                   - (1 if (column > game.width - 3 or row < 1) else 0)  # 2 o'clock blocked
                   - (1 if (column > game.width - 3 or row > game.height - 2) else 0)  # 4 o'clock blocked
                   - (1 if (column > game.width - 2 or row > game.height - 3) else 0)  # 5 o'clock bocked
                   - (1 if (column < 1 or row > game.height - 3) else 0)  # 7 o'clock blocked
                   - (1 if (column < 2 or row > game.height - 2) else 0)  # 8 o'clock blocked
                   )
    return total_moves


def legal_move_primary(game, player):
    if len(game.get_legal_moves(game.active_player)) == 0:
        return float('-inf') if player == game.active_player else float('inf')
    else:
        own_legal_moves = len(game.get_legal_moves(player=player))
        opponent_legal_moves = len(game.get_legal_moves(player=game.get_opponent(player=player)))
        return float(own_legal_moves - opponent_legal_moves)

def legal_move_primary_opp11(game, player):
    if len(game.get_legal_moves(game.active_player)) == 0:
        return float('-inf') if player == game.active_player else float('inf')
    else:
        own_legal_moves = len(game.get_legal_moves(player=player))
        opponent_legal_moves = len(game.get_legal_moves(player=game.get_opponent(player=player)))
        return float(own_legal_moves - 1.1*opponent_legal_moves)

def legal_move_primary_opp12(game, player):
    if len(game.get_legal_moves(game.active_player)) == 0:
        return float('-inf') if player == game.active_player else float('inf')
    else:
        own_legal_moves = len(game.get_legal_moves(player=player))
        opponent_legal_moves = len(game.get_legal_moves(player=game.get_opponent(player=player)))
        return float(own_legal_moves - 1.2*opponent_legal_moves)

def legal_move_primary_opp08(game, player):
    if len(game.get_legal_moves(game.active_player)) == 0:
        return float('-inf') if player == game.active_player else float('inf')
    else:
        own_legal_moves = len(game.get_legal_moves(player=player))
        opponent_legal_moves = len(game.get_legal_moves(player=game.get_opponent(player=player)))
        return float(own_legal_moves - .8*opponent_legal_moves)

def legal_move_primary_opp09(game, player):
    if len(game.get_legal_moves(game.active_player)) == 0:
        return float('-inf') if player == game.active_player else float('inf')
    else:
        own_legal_moves = len(game.get_legal_moves(player=player))
        opponent_legal_moves = len(game.get_legal_moves(player=game.get_opponent(player=player)))
        return float(own_legal_moves - .9*opponent_legal_moves)

def legal_move_primary_relsum(game, player):
    if len(game.get_legal_moves(game.active_player)) == 0:
        return float('-inf') if player == game.active_player else float('inf')
    else:
        own_legal_moves = len(game.get_legal_moves(player=player))
        opponent_legal_moves = len(game.get_legal_moves(player=game.get_opponent(player=player)))
        return float(own_legal_moves - opponent_legal_moves)/float(own_legal_moves + opponent_legal_moves)

def legal_move_primary_relmax(game, player):
    if len(game.get_legal_moves(game.active_player)) == 0:
        return float('-inf') if player == game.active_player else float('inf')
    else:
        own_legal_moves = len(game.get_legal_moves(player=player))
        opponent_legal_moves = len(game.get_legal_moves(player=game.get_opponent(player=player)))
        return float(own_legal_moves - opponent_legal_moves)/max(own_legal_moves, opponent_legal_moves)


def legal_move_relative(game, player):
        own_legal_moves = len(game.get_legal_moves(player=player))
        own_total_moves = possible_moves_count(*game.get_player_location(player), game)
        opponent_legal_moves = len(game.get_legal_moves(player=game.get_opponent(player=player)))
        opponent_total_moves = possible_moves_count(*game.get_player_location(player=game.get_opponent(player=player)), game)
        return own_legal_moves/own_total_moves - opponent_legal_moves/opponent_total_moves

def can_be_blocked(game, player):
    if game.active_player == player:
        afraid_of_blocking = 1
    else:
        afraid_of_blocking = -1

    return afraid_of_blocking * float(moves_intersect(game))


def norm_center_distance(game, player):
    center_row = (float(game.height) - 1.) / 2.
    center_col = (float(game.width) - 1.) / 2.
    max_distance = center_row ** 2 + center_col ** 2
    player_row, player_col = game.get_player_location(player=player)
    oppo_row, oppo_col = game.get_player_location(player=game.get_opponent(player=player))
    return (
            (oppo_row-center_row) ** 2 + (oppo_col-center_col) ** 2 -
            (player_row-center_row) ** 2 - (player_col-center_col) ** 2
            ) / max_distance


def custom_score(game, player: 'IsolationPlayer') ->float:

    return legal_move_primary(game, player)


def custom_score_2(game, player):

    #return legal_move_primary_relsum(game, player)
    return legal_move_primary_opp11(game, player)

def custom_score_3(game, player):
    return legal_move_primary_opp12(game, player)
#    return legal_move_primary(game, player) + moves_intersect(game=game)

class IsolationPlayer:

    def __init__(self, search_depth=100, score_fn=custom_score, timeout=10., name=None):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout
        self.name = name


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """
    reached_depth = 0

    def get_move(self, game, time_left, print_score=False):

        self.reached_depth = 0
        self.time_left = time_left
        best_move = (-1, -1)
        depth = 1
        try:
            while depth <= self.search_depth and depth <= self.reached_depth + 2:
                best_move = self.minimax(game, depth, print_score)
                depth += 1

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move, depth

    def minimax(self, game, depth, print_score):
        max_score, selected_move, score_list = self.my_minimax(game, depth, True)
        if print_score:
            print('Minimax with depth {} recommends move {} with score {}. Trace: {}'
                  .format(depth, selected_move, max_score, score_list))
        return selected_move


    def my_minimax(self, game, maxdepth, maximizer, depth=0, score_evals=None):
        if not score_evals:
            score_evals=list()
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if depth >= maxdepth:
            self.reached_depth = max(self.reached_depth, depth)
            finscore = self.score(game, self)
            print(finscore, score_evals)
            return finscore, (-1, -1), list()

        available_my_moves = game.get_legal_moves()
        if len(available_my_moves) == 0:
            finscore = self.score(game, self)
            print(finscore, score_evals)
            self.reached_depth = max(self.reached_depth, depth)
            return finscore, (-1, -1), list()

        score_list = list()

        next_move = (-1, -1)
        if maximizer == True:  # Maximizing player
            maximizer = False
            utility_score = float("-inf")
            for move in available_my_moves:
                new_game = game.forecast_move(move)
                next_score, _, __ = self.my_minimax(new_game, maxdepth, maximizer, depth=depth+1,
                                                score_evals=score_evals + [move])
                if utility_score < next_score or utility_score == float("-inf"):
                    utility_score = next_score
                    next_move = move
                score_list.append((move, next_score))

        else:
            maximizer = True
            utility_score = float("inf")
            for move in available_my_moves:
                new_game = game.forecast_move(move)
                next_score, _, __ = self.my_minimax(new_game, maxdepth, maximizer, depth=depth+1,
                                                score_evals=score_evals + [move])
                if utility_score > next_score or utility_score == float("inf"):
                    utility_score = next_score
                    next_move = move

        return utility_score, next_move, score_list

    def minimax2(self, game, depth: int) -> (int, int):

        def max_move(game, max_depth, currdepth=0):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if max_depth > currdepth:
                score = float('-inf')
                legal_moves = game.get_legal_moves()
                for legal_move in legal_moves:
                    score = max(score, min_move(game.forecast_move(move=legal_move),
                                                max_depth=max_depth,
                                                currdepth=currdepth+1))
                return score
            else:
                return self.score(game, self)



        def min_move(game, max_depth, currdepth=0):
            if self.time_left() < self.TIMER_THRESHOLD:
                raise SearchTimeout()
            if max_depth > currdepth:
                score = float('inf')
                legal_moves = game.get_legal_moves()
                for legal_move in legal_moves:
                    score = min(score, max_move(game.forecast_move(move=legal_move),
                                                max_depth=max_depth,
                                                currdepth=currdepth+1))
                return score
            else:
                return self.score(game, self)

        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()
        depth = depth - 1
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return (-1, -1)

        max_score = float('-inf')
        recommended_move = -1, -1
        for legal_move in legal_moves:
            score = min_move(game.forecast_move(move=legal_move), max_depth=depth)
            if score >= max_score:
                recommended_move = legal_move
                max_score = score
        return recommended_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    reached_depth = 0

    def get_move(self, game, time_left, print_score=False):
        self.time_left = lambda: time_left() - self.TIMER_THRESHOLD

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout

        self.reached_depth = 0
        depth = 1
        return_move = (-1, -1)
        while depth <= self.reached_depth + 2:
            self.reached_depth = 0
            try:
                # The try/except block will automatically catch the exception
                # raised when the timer is about to expire.
                move = self.alphabeta(game, depth, print_score=print_score)
                if move == (-1, -1):
                    #print('Existential Crisis after {} moves with {}ms left'.format(search_depth, self.time_left()))
                    break
                    #return return_move
                #elif will_win:
                #    return_move = move
                #    return return_move
                else:
                    return_move = move
                depth += 1

            except SearchTimeout:
                #print('timeing out at search depth {} at time {}'.format(search_depth - 1, self.time_left()))
                return return_move, depth
            except FoundWinningMoveException as fwme:
                return fwme.move, depth

        # Return the best move from the last completed search iteration
        return return_move, depth - 1

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), print_score=False):

        selected_move = (-1, -1)
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            return selected_move
        max_score = float('-inf')
        score_list=list()
        for legal_move in legal_moves:
            score = self.recursive_alphabeta(game=game.forecast_move(legal_move),
                                             depth=1,
                                             max_depth=depth,
                                             alpha=max_score,
                                             beta=beta,
                                             is_max=False)
            if score > max_score or (max_score == float('-inf')):
                max_score = score
                selected_move = legal_move
            score_list.append((legal_move, score))
        if print_score:
            print('AlphaBeta with depth {} recommends move {} with score {}. Trace: {}'
                  .format(depth, selected_move, max_score, score_list))
        if max_score == float('inf'):
            raise FoundWinningMoveException(move=selected_move)
        return selected_move#, max_score == float('inf')

    def recursive_alphabeta(self, game, depth, max_depth, alpha, beta, is_max):
        if self.time_left() < 0:
            # print('Raising SearchTimeout, time left:{}'.format(self.time_left()))
            raise SearchTimeout
        if depth >= max_depth:
            self.reached_depth = max(self.reached_depth, depth)
            return self.score(game, self)

        legal_moves = game.get_legal_moves()
        if not legal_moves:
            self.reached_depth = max(self.reached_depth, depth)
            return float('-inf') if is_max else float('inf')

        running_score = float('inf') * ((-1) ** float(is_max))
        for legal_move in legal_moves:
            recursive_score = self.recursive_alphabeta(game=game.forecast_move(legal_move),
                                                       depth=depth + 1,
                                                       max_depth=max_depth,
                                                       alpha=alpha,
                                                       beta=beta,
                                                       is_max=(not is_max))
            if is_max:
                running_score = max(running_score, recursive_score)
                alpha = max(alpha, running_score)
            else:
                running_score = min(running_score, recursive_score)
                beta = min(beta, running_score)
            if beta <= alpha:
                break
        self.reached_depth = max(self.reached_depth, depth)
        return running_score

