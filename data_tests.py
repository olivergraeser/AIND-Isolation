import sys
import itertools
import random
import json

from collections import namedtuple, defaultdict

from isolation import Board
from sample_players import (RandomPlayer, open_move_score,
                            improved_score, center_score)
from game_agent import (MinimaxPlayer, AlphaBetaPlayer, legal_move_primary, legal_move_primary_relmax,
                        legal_move_primary_relsum, legal_move_primary_opp13, legal_move_primary_opp14,
                        legal_move_primary_opp11, legal_move_primary_opp12, legal_move_primary_opp07,
                        legal_move_primary_opp08, legal_move_primary_opp09, legal_move_primary_opp06,
                        peekaboo, legal_primary_center)

player_types = {'random': RandomPlayer,
                'minimax': MinimaxPlayer,
                'alphabeta': AlphaBetaPlayer}

score_functions = {'open': open_move_score,
                   'center': center_score,
                   'improved': improved_score,
                   'legal_move_primary':legal_move_primary,
                   'legal_move_primary_relmax':legal_move_primary_relmax,
                   'legal_move_primary_relsum':legal_move_primary_relsum,
                   'legal_move_primary_opp13': legal_move_primary_opp13,
                   'legal_move_primary_opp14': legal_move_primary_opp14,
                   'legal_move_primary_opp11': legal_move_primary_opp11,
                   'legal_move_primary_opp12': legal_move_primary_opp12,
                   'legal_move_primary_opp07': legal_move_primary_opp07,
                   'legal_move_primary_opp08': legal_move_primary_opp08,
                   'legal_move_primary_opp09': legal_move_primary_opp09,
                   'legal_move_primary_opp06': legal_move_primary_opp06,
                   'peekaboo': peekaboo,
                   'legal_primary_center': legal_primary_center
                   }

NUM_MATCHES = 5  # number of matches against each opponent
TIME_LIMIT = 150  # number of milliseconds before timeout
PLAYER_ONE_TYPE = None
PLAYER_TWO_TYPE = None
PLAYER_ONE_FUNCTION = None
PLAYER_TWO_FUNCTION = None
FILE_NAME = '??'

DESCRIPTION = """
This script evaluates the performance of the custom_score evaluation
function against a baseline agent using alpha-beta search and iterative
deepening (ID) called `AB_Improved`. The three `AB_Custom` agents use
ID and alpha-beta search with the custom_score functions defined in
game_agent.py.
"""

Agent = namedtuple("Agent", ["player", "name"])


def play_round(cpu_agent, test_agent, num_matches):
    win_count = dict()
    timeout_count = dict()
    forfeit_count = dict()
    game_data = list()
    win_count = defaultdict(int)
    timeout_count = defaultdict(int)
    forfeit_count = defaultdict(int)

    for _ in range(num_matches):

        games = [Board(cpu_agent.player, test_agent.player, record_game=True),
                      Board(test_agent.player, cpu_agent.player, record_game=True)]
        # initialize all games with a random move and response
        for _ in range(2):
            move = random.choice(games[0].get_legal_moves())
            for game in games:
                game.apply_move(move)

        # play all games and tally the results
        for game in games:
            winner, data, termination = game.play(time_limit=TIME_LIMIT)
            win_count[winner] += 1
            game_data.append(data)

            if termination == "timeout":
                timeout_count[winner] += 1
            elif termination == "forfeit":
                forfeit_count[winner] += 1

    return game_data, win_count, forfeit_count, timeout_count


def update(total_wins, wins):
    for player in total_wins:
        total_wins[player] += wins[player]
    return total_wins


def play_matches(own_agent, opponent_agent, num_matches):
    """Play matches between the test agent and each cpu_agent individually. """

    game_data, win_count, forfeit_count, timeout_count = play_round(own_agent, opponent_agent, num_matches)

    _total = 2 * num_matches

    relative_win_count = {agent.name: (win_count[agent.player], _total - win_count[agent.player],
                                        win_count[agent.player]/_total) for agent in [own_agent, opponent_agent] }


    print('Win ration for players:\n {}'.format('\n '.join(['{}: {}/{} -> {}'.format(key, *item)
                                                            for key, item in relative_win_count.items()])))
    return game_data



def main():
    own_id = '{}:{}'.format(PLAYER_ONE_FUNCTION, PLAYER_ONE_TYPE)
    own_agent = Agent(player_types[PLAYER_ONE_TYPE](score_fn=score_functions[PLAYER_ONE_FUNCTION],
                                                    name=own_id),
                      own_id)


    # Define a collection of agents to compete against the test agents

    opponent_id = '{}:{}'.format(PLAYER_TWO_FUNCTION, PLAYER_TWO_TYPE)
    opponent_agent = Agent(player_types[PLAYER_TWO_TYPE](score_fn=score_functions[PLAYER_TWO_FUNCTION],
                                                         name=opponent_id),
                           opponent_id)

    data = play_matches(own_agent, opponent_agent, NUM_MATCHES)
    file_name = '{}_{}_{}_{}_{}'.format(PLAYER_ONE_TYPE, PLAYER_ONE_FUNCTION, PLAYER_TWO_TYPE, PLAYER_TWO_FUNCTION, NUM_MATCHES)
    with open(file_name, 'w') as f:
        f.write(json.dumps(data))


if __name__ == "__main__":
    _, NUM_MATCHES, PLAYER_ONE_TYPE, PLAYER_ONE_FUNCTION, PLAYER_TWO_TYPE, PLAYER_TWO_FUNCTION = sys.argv
    NUM_MATCHES = int(NUM_MATCHES)
    main()
