################################################################################
# 6.0002 Fall 2021
# Problem Set 1
# Name: Autumn Artist
# Collaborators: Office Hours
# Time: 10 hours

from state import State

##########################################################################################################
## Problem 1
##########################################################################################################

def load_election(filename):
    """
    Reads the contents of a file, with data given in the following tab-separated format:
    State[tab]Democrat_votes[tab]Republican_votes[tab]EC_votes

    Please ignore the first line of the file, which are the column headers, and remember that
    the special character for tab is '\t'

    Parameters:
    filename - the name of the data file as a string

    Returns:
    a list of State instances
    """
    # TODO
    election_file = open(filename, 'r')
    election = []
    #first line - so read and dealt with
    line1 = election_file.readline()
    #goes through each line after the first
    for i in election_file:
        #splits the by tab
        state_info = i.split('\t')
        #since the ec has '\n' we need to get rid of it
        if "\n" in i:
            num = state_info[3]
            num = num.replace('\n', '')
            state = State(state_info[0], state_info[1], state_info[2], num)
        #for last line since it doesn't contain "\n"
        else:
            state = State(state_info[0], state_info[1], state_info[2], state_info[3])
        #adding each state objection in an array
        election.append(state)
    return election

##########################################################################################################
## Problem 2: Helper functions
##########################################################################################################

def election_winner(election):
    """
    Finds the winner of the election based on who has the most amount of EC votes.
    Note: In this simplified representation, all of EC votes from a state go
    to the party with the majority vote.

    Parameters:
    election - a list of State instances

    Returns:
    a tuple, (winner, loser) of the election i.e. ('dem', 'rep') if Democrats won, else ('rep', 'dem')
    """
    # TODO
    #counting the number of dem wins and rep wins by EC votes
    demcount = 0
    repcount = 0
    #goes through each state instance
    for i in election:
        #i is a state instance
        winner = i.get_winner()
        if winner == "dem":
            demcount += i.get_ecvotes() 
        else:
            repcount += i.get_ecvotes() 
    #tuple changes depending on who wins
    if demcount>repcount:
        return ("dem", "rep")
    else:
        return ("rep", "dem")


def winner_states(election):
    """
    Finds the list of States that were won by the winning candidate (lost by the losing candidate).

    Parameters:
    election - a list of State instances

    Returns:
    A list of State instances won by the winning candidate
    """
    # finds the winner
    winner = election_winner(election)
    #list for won states
    won_states = []
    for i in election:
        #if the winner is the same as the overall winner
        if i.get_winner() == winner[0]:
            #list of state instances
            won_states.append(i)
    return won_states


def ec_votes_to_flip(election, total=538):
    """
    Finds the number of additional EC votes required by the loser to change election outcome.
    Note: A party wins when they earn half the total number of EC votes plus 1.

    Parameters:
    election - a list of State instances
    total - total possible number of EC votes

    Returns:
    int, number of additional EC votes required by the loser to change the election outcome
    """
    # TODO
    #to win you need half the total +1
    winning_votes_needed = int(total/2+1)
    #find number of EC votes the loser has
    ec_loser_votes = 0
    for i in election:
        if i not in winner_states(election):
            ec_loser_votes += i.get_ecvotes()
    # need >= winning_vote_needed
    return winning_votes_needed-ec_loser_votes


##########################################################################################################
## Problem 3: Brute Force approach
##########################################################################################################

def combinations(L):
    """
    Helper function to generate powerset of all possible combinations
    of items in input list L. E.g., if
    L is [1, 2] it will return a list with elements
    [], [1], [2], and [1,2].

    Parameters:
    L - list of items

    Returns:
    a list of lists that contains all possible
    combinations of the elements of L
    """

    def get_binary_representation(n, num_digits):
        """
        Inner function to get a binary representation of items to add to a subset,
        which combinations() uses to construct and append another item to the powerset.

        Parameters:
        n and num_digits are non-negative ints

        Returns:
            a num_digits str that is a binary representation of n
        """
        result = ''
        while n > 0:
            result = str(n%2) + result
            n = n//2
        if len(result) > num_digits:
            raise ValueError('not enough digits')
        for i in range(num_digits - len(result)):
            result = '0' + result
        return result

    powerset = []
    for i in range(0, 2**len(L)):
        binStr = get_binary_representation(i, len(L))
        subset = []
        for j in range(len(L)):
            if binStr[j] == '1':
                subset.append(L[j])
        powerset.append(subset)
    return powerset

def brute_force_swing_states(winner_states, ec_votes):
    """
    Finds a subset of winner_states that would change an election outcome if
    voters moved into those states, these are our swing states. Iterate over
    all possible move combinations using the helper function combinations(L).
    Return the move combination that minimises the number of voters moved. If
    there exists more than one combination that minimises this, return any one of them.

    Parameters:
    winner_states - a list of State instances that were won by the winner
    ec_votes - int, number of EC votes needed to change the election outcome

    Returns:
    A list of State instances such that the election outcome would change if additional
    voters relocated to those states
    The empty list, if no possible swing states
    """
    # TODO

    final_combo = []
    minvoters = None
    #iterate through each combination
    for i in combinations(winner_states):
        ec_vote_sum = 0
        moved_voters = 0
        #in case the combinations have more than one state instance
        for j in i:
            ec_vote_sum += j.get_ecvotes()
            moved_voters += (j.get_margin()+1)
        if ec_vote_sum>= ec_votes:
            if minvoters == None or moved_voters<minvoters:
                minvoters = moved_voters
                final_combo = i
    return final_combo
    


##########################################################################################################
## Problem 4: Dynamic Programming
## In this section we will define two functions, move_max_voters and move_min_voters, that
## together will provide a dynamic programming approach to find swing states. This problem
## is analagous to the complementary knapsack problem, you might find Lecture 1 of 6.0002 useful
## for this section of the pset.
##########################################################################################################

def move_max_voters(winner_states, ec_votes, states = None):
    """
    Finds the largest number of voters needed to relocate to get at most ec_votes
    for the election loser.

    Analogy to the knapsack problem:
        Given a list of states each with a weight(ec_votes) and value(margin+1),
        determine the states to include in a collection so the total weight(ec_votes)
        is less than or equal to the given limit(ec_votes) and the total value(voters displaced)
        is as large as possible.

    Parameters:
    winner_states - a list of State instances that were won by the winner
    ec_votes - int, the maximum number of EC votes

    Returns:
    A list of State instances such that the maximum number of voters need to be relocated
    to these states in order to get at most ec_votes
    The empty list, if every state has a # EC votes greater than ec_votes
    """
    # TODO
    if states == None:
        states = {}
    if (len(winner_states), ec_votes) in states:
        return  states[(len(winner_states), ec_votes)]
    
    if winner_states == [] or ec_votes == 0:
        swings = []
    elif winner_states[0].get_ecvotes() > ec_votes:
        #if the winner state is over capacity
        swings = move_max_voters(winner_states[1:], ec_votes, states)
    else:
        next_swing = winner_states[0]
        #Tests what would happen if you addded the state
        add_state = move_max_voters(winner_states[1:], ec_votes-next_swing.get_ecvotes(), states) 
        add_vote = 0
        for i in add_state:
            add_vote += i.get_margin()+1
        add_vote += next_swing.get_margin()
        #Check what would happen if you didn't add the state
        no_state = move_max_voters(winner_states[1:], ec_votes, states)
        no_vote = 0
        for i in no_state:
            no_vote += i.get_margin()+1
        #which is better
        if add_vote > no_vote:
            swings = add_state+[next_swing]
        else:
            swings = no_state
            
    states[(len(winner_states), ec_votes)] = swings
    return swings



def move_min_voters(winner_states, ec_votes_needed):
    """
    Finds a subset of winner_states that would change an election outcome if
    voters moved into those states. Should minimize the number of voters being relocated.
    Only return states that were originally won by the winner (lost by the loser)
    of the election.

    Hint: This problem is simply the complement of move_max_voters. You should call
    move_max_voters with ec_votes set to (#ec votes won by original winner - ec_votes_needed)

    Parameters:
    winner_states - a list of State instances that were won by the winner
    ec_votes_needed - int, number of EC votes needed to change the election outcome

    Returns:
    A list of State instances such that the election outcome would change if additional
    voters relocated to those states (also can be referred to as our swing states)
    The empty list, if no possible swing states
    """
    # TODO
    #finds the number of ec votes won by winner
    ec_won = 0
    for i in winner_states:
        ec_won += i.get_ecvotes()
    #list of states for winner
    max_voter_states = move_max_voters(winner_states, ec_won-ec_votes_needed)
    min_states = []
    #if the state is in the winner states but not in max states
    for i in winner_states:
        if i not in max_voter_states:
            min_states.append(i)
    return min_states


##########################################################################################################
## Problem 5
##########################################################################################################

def relocate_voters(election, swing_states, states_with_pride = ['AL', 'AZ', 'CA', 'TX']):
    """
    Finds a way to shuffle voters in order to flip an election outcome. Moves voters
    from states that were won by the losing candidate (states not in winner_states), to
    each of the states in swing_states. To win a swing state, you must move (margin + 1)
    new voters into that state. Any state that voters are moved from should still be won
    by the loser even after voters are moved. Also finds the number of EC votes gained by
    this rearrangement, as well as the minimum number of voters that need to be moved.
    Note: You cannot move voters out of Alabama, Arizona, California, or Texas.

    Parameters:
    election - a list of State instances representing the election
    swing_states - a list of State instances where people need to move to flip the election outcome
                   (result of move_min_voters or greedy_swing_states)
    states_with_pride - a list of Strings holding the names of states where residents cannot be moved from
                    (default states are AL, AZ, CA, TX)

    Return:
    A tuple that has 3 elements in the following order:
        - an int, the total number of voters moved
        - a dictionary with the following (key, value) mapping:
            - Key: a 2 element tuple of str, (from_state, to_state), the 2 letter State names
            - Value: int, number of people that are being moved
        - an int, the total number of EC votes gained by moving the voters
    None, if it is not possible to sway the election
    """
    # TODO
    #finding losing candidate states without a lot of pride
    loser_states= []
    for i in election:
        if i not in winner_states(election) and i.get_name() not in states_with_pride:
            loser_states.append(i)
    #Finding ec votes needed to flip election
    ec_votes_needed = ec_votes_to_flip(election)
    #find minimum swingers needed to flip election
    swingers = move_min_voters(swing_states, ec_votes_needed)
    
    #things to keep track of
    moved_states = {}
    voters_moved = 0
    ec_voters_gained = 0
    
    #gonna be removing states as we use them
    while len(loser_states) != 0 and len(swingers) != 0:
        swing = swingers[-1]
        loser = loser_states[-1]
        loser_margin = loser.get_margin()-1
        swing_margin = swing.get_margin()+1
        #if the loser state can lose rid of more voters than the swinger need to gain
        if loser_margin > swing_margin:
            #moving the voters
            swing.add_losing_candidate_voters(swing_margin)
            loser.subtract_winning_candidate_voters(swing_margin)
            #added the voters to moved
            voters_moved += swing_margin
            ec_voters_gained += swing.get_ecvotes()
            swingers.pop()
            #dictionary to keep track of where each person moved
            moved_states[(loser.get_name(), swing.get_name())] = swing_margin
        else:
            #if the loser state can lose less than are needed to flip the swing
            swing.add_losing_candidate_voters(loser_margin)
            loser.subtract_winning_candidate_voters(loser_margin)
            voters_moved += loser_margin
            loser_states.pop()
            moved_states[(loser.get_name(), swing.get_name())] = loser_margin
    #if there are still swing states left after using all the loser states
    if len(swingers) != 0:
        return None
    else:
        return (voters_moved, moved_states, ec_voters_gained)
                   



if __name__ == "__main__":
    pass
    #election_sample = load_election("600_results.txt")
    #lost_states_sample, ec_needed_sample = winner_states(election_sample), ec_votes_to_flip(election_sample)
    #swing_states_sample = move_min_voters(lost_states_sample, ec_needed_sample)
    #results_sample_dp = relocate_voters(election_sample, swing_states_sample)
    #print(results_sample_dp)


    # Uncomment the following lines to test each of the problems

    # # tests Problem 1
    #year = 2012
    #election = load_election("%s_results.txt" % year)
    #print(len(election))
    #print(election[0])

    # # tests Problem 2
    #winner, loser = election_winner(election)
    #won_states = winner_states(election)
    #names_won_states = [state.get_name() for state in won_states]
    #reqd_ec_votes = ec_votes_to_flip(election)
    #print("Winner:", winner, "\nLoser:", loser)
    #print("States won by the winner: ", names_won_states)
    #print("EC votes needed:",reqd_ec_votes, "\n")

    # # tests Problem 3

    #brute_election = load_election("60002_results.txt")
    #brute_won_states = winner_states(brute_election)
    #brute_ec_votes_to_flip = ec_votes_to_flip(brute_election, total=14)
    #brute_swing = brute_force_swing_states(brute_won_states, brute_ec_votes_to_flip)
    #names_brute_swing = [state.get_name() for state in brute_swing]
    #voters_brute = sum([state.get_margin()+1 for state in brute_swing])
    #ecvotes_brute = sum([state.get_ecvotes() for state in brute_swing])
    #print("Brute force swing states results:", names_brute_swing)
    #print("Brute force voters displaced:", voters_brute, "for a total of", ecvotes_brute, "Electoral College votes.\n")

    
    #results_sample = [state.get_name() for state in results_sample_list]
    # # tests Problem 4: move_max_voters
    #print("move_max_voters")
    #total_lost = sum(state.get_ecvotes() for state in won_states)
    #non_swing_states = move_max_voters(won_states, total_lost-reqd_ec_votes)
    #non_swing_states_names = [state.get_name() for state in non_swing_states]
    #max_voters_displaced = sum([state.get_margin()+1 for state in non_swing_states])
    #max_ec_votes = sum([state.get_ecvotes() for state in non_swing_states])
    #print("States with the largest margins (non-swing states):", non_swing_states_names)
    #print("Max voters displaced:", max_voters_displaced, "for a total of", max_ec_votes, "Electoral College votes.", "\n")

    # # tests Problem 4: move_min_voters
    # print("move_min_voters")
    #swing_states = move_min_voters(won_states, reqd_ec_votes)
    # swing_state_names = [state.get_name() for state in swing_states]
    # min_voters_displaced = sum([state.get_margin()+1 for state in swing_states])
    # swing_ec_votes = sum([state.get_ecvotes() for state in swing_states])
    # print("Complementary knapsack swing states results:", swing_state_names)
    # print("Min voters displaced:", min_voters_displaced, "for a total of", swing_ec_votes, "Electoral College votes. \n")

    # # tests Problem 5: relocate_voters
    #print("relocate_voters")
    #flipped_election = relocate_voters(election, swing_states)
    #print("Flip election mapping:", flipped_election)