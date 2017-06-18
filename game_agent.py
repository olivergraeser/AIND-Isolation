"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random



def open_move_score(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return float(len(game.get_legal_moves(player)))


def improved_score(game, player):

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)


def center_score(game, player):
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)
    return float((h - y)**2 + (w - x)**2)


class FoundWinningMoveException(Exception):
    def __init__(self, move):
        self.move = move
class ExistentialCrisisException(Exception):
    def __init__(self, move, score):
        self.move = move
        self.score = score

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
    return bool(set(p1moves) & set(p2moves))


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

def legal_move_primary_opp13(game, player):
    if len(game.get_legal_moves(game.active_player)) == 0:
        return float('-inf') if player == game.active_player else float('inf')
    else:
        own_legal_moves = len(game.get_legal_moves(player=player))
        opponent_legal_moves = len(game.get_legal_moves(player=game.get_opponent(player=player)))
        return float(own_legal_moves - 1.3*opponent_legal_moves)

def legal_move_primary_opp14(game, player):
    if len(game.get_legal_moves(game.active_player)) == 0:
        return float('-inf') if player == game.active_player else float('inf')
    else:
        own_legal_moves = len(game.get_legal_moves(player=player))
        opponent_legal_moves = len(game.get_legal_moves(player=game.get_opponent(player=player)))
        return float(own_legal_moves - 1.4*opponent_legal_moves)


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

    return improved_score(game, player)
    return legal_move_primary(game, player)


def custom_score_2(game, player):

    return center_score(game, player)
    return legal_move_primary_relmax(game, player)


def custom_score_3(game, player):

    return open_move_score(game, player)
    return legal_move_primary_relsum(game, player)

class IsolationPlayer:

    def __init__(self, search_depth=100, score_fn=custom_score, timeout=10., name=None, ignore_timeout=False):
        self.ignore_timeout = ignore_timeout
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

    def __init__(self, search_depth=100, score_fn=custom_score, timeout=10., record_tree=False):
        self.record_tree = record_tree
        self.search_information = dict()
        super().__init__(search_depth=search_depth, score_fn=score_fn, timeout=timeout)

    def get_move(self, game, time_left):
        self.search_information = dict()
        self.time_left = lambda: time_left() - self.TIMER_THRESHOLD

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        move = (-1, -1)
        score = float('-inf')
        search_depth = 1
        try:
            while search_depth <= self.search_depth:
                self.search_information[search_depth] = (move, score)
                move, score = self.recursive_minimax(game, search_depth, True)
                search_depth += 1

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return move

    def minimax(self, game, depth):
        move, _ = self.recursive_minimax(game, depth, True)
        return move


    def recursive_minimax(self, game, depth, is_max):
        if self.time_left() < 0:
            raise SearchTimeout()

        if depth <= 0:
            running_score, best_move = self.score(game, self), (-1, -1)
        else:
            legal_moves = game.get_legal_moves()
            if len(legal_moves) == 0:
                running_score, best_move = self.score(game, self), (-1, -1)
            else:
                best_move = (-1, -1)
                if is_max:  # Maximizing player
                    is_max = False
                    running_score = float("-inf")
                    for legal_move in legal_moves:
                        new_game = game.forecast_move(legal_move)
                        _, score = self.recursive_minimax(new_game, depth - 1, is_max)
                        if running_score <= score:
                            running_score = score
                            best_move = legal_move
                else:
                    is_max = True
                    running_score = float("inf")
                    for legal_move in legal_moves:
                        new_game = game.forecast_move(legal_move)
                        _, score = self.recursive_minimax(new_game, depth - 1, is_max)
                        if running_score >= score:
                            running_score = score
                            best_move = legal_move
        return best_move, running_score


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def __init__(self, search_depth=100, score_fn=custom_score, timeout=10., name=None, record_tree=False):
        self.record_tree = record_tree
        self.search_information = dict()
        super().__init__(search_depth=search_depth, score_fn=score_fn, timeout=timeout, name=name)

    def reset_for_new_game(self):
        self.search_information = dict()

    def get_move(self, game, time_left):
        self.time_left = lambda: time_left() - self.TIMER_THRESHOLD
        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout

        search_depth = 1
        return_move = (-1, -1)
        score = float('-inf')
        search_history = dict()
        count = 0
        try:
            while search_depth <= self.search_depth:
                search_history[search_depth] = (return_move, score)

                # The try/except block will automatically catch the exception
                # raised when the timer is about to expire.
                if self.record_tree:
                    move, score, count = self.rec_alphabeta(game, search_depth)
                    #count: number of available moves right now
                else:
                    move = self.alphabeta(game, search_depth)

                return_move = move

                search_depth += 1

        except SearchTimeout:
            #timeout - return best_move of last completed round
            pass
        except FoundWinningMoveException as fwme:
            #prevents the agent from searching thousands of levels deep if the agent knows it will already win
            #after three rounds
            return_move = fwme.move
            search_history[search_depth+1] = (return_move, float('inf'))
        except ExistentialCrisisException as ece:
            #prevents the agent from searching thousands of levels deep if the agent knows it will already win
            #after three rounds
            return_move = ece.move
            search_history[search_depth+1] = (return_move, float('-inf'))


        # Return the best move from the last completed search iteration
        if self.record_tree:
            search_history['cnt'] = count
            self.search_information[game.get_movecount() + 1] = search_history

        #print(time_left())
        return return_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        move, score, count = self.rec_alphabeta(game=game, depth=depth, alpha=alpha, beta=beta)
        return move

    def rec_alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):

        selected_move = (-1, -1)
        legal_moves = game.get_legal_moves()
        move_count = 0 if not self.record_tree else len(legal_moves)
        max_score = float('-inf')
        if not legal_moves:
            raise ExistentialCrisisException(move=selected_move, score=max_score)
        else:
            for legal_move in legal_moves:
                score = self.recursive_alphabeta(game=game.forecast_move(legal_move),
                                                 depth=depth,
                                                 alpha=max_score,
                                                 beta=beta,
                                                 is_max=False)
                if score > max_score or (max_score == float('-inf')):
                    max_score = score
                    selected_move = legal_move
            if max_score == float('inf'):
                #The move I found will guarantee me to win. I do not need to search deeper
                raise FoundWinningMoveException(move=selected_move)
            elif max_score == float('-inf'):
                raise ExistentialCrisisException(move=selected_move, score=max_score)
        return selected_move, max_score, move_count

    def recursive_alphabeta(self, game, depth, alpha, beta, is_max):
        if self.time_left() < 0:
            raise SearchTimeout
        if depth <= 1:
            running_score = self.score(game, self)
        else:
            legal_moves = game.get_legal_moves()
            if not legal_moves:
                running_score = float('-inf') if is_max else float('inf')
            else:
                running_score = float('inf') * ((-1) ** float(is_max))
                for legal_move in legal_moves:
                    recursive_score = self.recursive_alphabeta(game=game.forecast_move(legal_move),
                                                               depth=depth - 1,
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
        return running_score
