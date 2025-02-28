# -------------------------------------------------------------------------------


# King's College London - 6CCS3AIN Coursework 26th November 2024
# Niki Norgren, niki.norgren@kcl.ac.uk

# Code references: 
# Parts of class Grid used from mapAgents.py 
# Code for calculating the Expected Utility adapted from Week 4 lecture slides
# Implementation ideas for Value Iteration adapted from AIMA: Stuart Russel, 
# Peter Norvig (2022), Chapter 16: Pages 562-565.
# Parts of the Reward Model adapted from Week 6 + 7 Practical Sheet

# -------------------------------------------------------------------------------

# An implementation of a Markov Decision Process, utilising value iteration.


from pacman import Directions
from game import Agent
import api  
import copy
import numbers


class MDPAgent(Agent):

    # constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        print "Starting up MDPAgent!"
        name = "Pacman"

    def registerInitialState(self, state):
        self.corners = api.corners(state)
        self.height = max(self.corners, key=lambda x: x[1])[1] + 1  # +1 to adjust for 0 based indexing
        self.width = max(self.corners)[0] + 1  
        self.walls = api.walls(state)
    
    def final(self, state):
        print "Looks like the game just ended!"

    # the next possible coordinates for an agent
    def get_possible_coord(self, (x, y)):

        return [(x - 1, y), # west
                (x, y + 1), # north
                (x, y - 1), # south
                (x + 1, y)] # east

    # calculate the expected utility of a state
    def expected_utility(self, state, grid, row, col):

        directions = [(0, -1, Directions.NORTH), 
                      (1, 0, Directions.EAST),   
                      (0, 1, Directions.SOUTH),  
                      (-1, 0, Directions.WEST)] 

        # the possible neighbouring positions in directions
        coord = [(col + dx, row + dy) for dx, dy, i in directions]
        # each direction is assigned an initial probability of 0
        prob = [(0, direction) for i, j, direction in directions]

        # replace the wall position with the current position
        coord = [((col, row) if not isinstance(grid[r, c], numbers.Number) else (c, r),
        grid[(row if not isinstance(grid[r, c], numbers.Number) else r), 
             (col if not isinstance(grid[r, c], numbers.Number) else c)]) for c, r in coord]

        # identify coordinates that are perpendicular to the current coordinate
        # and multiply by correct utility of that state
        num_coords = len(coord)
        for i, (x, y) in enumerate(coord):
            perpendicular1 = (i - 1) % num_coords
            perpendicular2 = (i + 1) % num_coords

            prob[i] = (0.8 * coord[i][1] +
                       0.1 * coord[perpendicular1][1] +
                       0.1 * coord[perpendicular2][1],
                       prob[i][1])
        return prob


    # valute iteration is performed where a grid is returned with the updated utility values
    def value_iteration(self, state, grid):

        grid_copy = copy.deepcopy(grid)
        threshold = 0.1     # convergence threshold
        iterations = 30     # number of iterations
        gamma = 0.9         # discount factor   

       
        food = api.food(state)
        capsules = api.capsules(state)
        # convert ghost coordinates to ints (as api.py uses floats)
        ghosts = [(int(x), int(y)) for x, y in api.ghosts(state)]
        
        # these values do not need to be iterated over
        uniterated = api.walls(state) + api.ghosts(state)

        food_reward = 10 
        ghost_reward = 1

        grid = Grid(self.width, self.height, -0.07)        # initialise the grid with a negative reward
        grid.set_grid_values(capsules, 0.8 * food_reward)  # capsule reward
        grid.set_grid_values(food, 1 * food_reward)        # food reward
        grid.set_grid_values(ghosts, -10 * ghost_reward)   # positions of ghosts have negative values

        while iterations > 0: 
            U = copy.deepcopy(grid_copy) # a copy of the old grid 
            difference = 0               # the difference between iterations on the grid

            for x in range(self.height):
                for y in range(self.width):
                    value = grid_copy[x, y]
                    # check to make sure the position is not where the ghost or walls are
                    if (y, grid.column(x)) not in uniterated:
                        # use the list returned from expected_utility
                        expected_utility = [utility[0] for utility in self.expected_utility(state, U, x, y)]
                        MEU = max(expected_utility) # the maximum expected utility 
                        # perform Bellman's equation and update the values
                        grid_copy[x, y] = grid[x, y] + gamma * MEU 

                # reward model for proximity to ghosts
                for ghost in ghosts: 
                    # keep track of potential positions of ghosts as they move currently
                    for position in self.get_possible_coord(ghost):
                        if position not in uniterated:
                            # reward pacman negatively for positions of the ghosts
                            grid[int(grid.column(position[1])), int(position[0])] = -6 * ghost_reward
                            for position2 in self.get_possible_coord(position):
                                # reward pacman negatively for positions of the ghosts by two moves
                                if position2 not in uniterated:
                                    grid[int(grid.column(position2[1])), int(position2[0])] = -6 * ghost_reward

            # the difference in utility for every coordinate of the old grid and the new grid
            for x in range(self.height):
                for y in range(self.width):
                    if (y, grid.column(x)) not in uniterated: 
                        value = grid_copy[x, y]
                        difference += abs(round(value - U[x, y], 4))

            if difference <= threshold: # convergence check: if the total difference in utility across the grid is less than or equal to threshold we stop
                break

            iterations -= 1

        return grid_copy
    

    # retrieves the action for pacman to take based on value iteration
    def getAction(self, state):

        current_pos = api.whereAmI(state)
        legal = api.legalActions(state)

        # perform value iteration
        grid = Grid(self.width, self.height, -0.07)
        grid = self.value_iteration(state, grid)

        # calculate expected utility for each legal action
        expected_utilities = self.expected_utility(state, grid, grid.column(current_pos[1]), current_pos[0])

        # select the action with the maximum expected utility
        best_action = max([(utility, action) for utility, action in expected_utilities if action in legal], key=lambda x: x[0])[1]
        return api.makeMove(best_action, legal)
    

class Grid:
    # grid: an array that has one position for each element in the grid
    # width: the width of the grid
    # height: the height of the grid
    # init_reward: initialises the grid with default reward values

    def __init__(self, width, height, init_reward):
        
        self.height = height
        self.width = width

        self.grid = []
        for i in range(self.height):
            j = [init_reward] * self.width
            self.grid.append(j)

    def get_grid_width(self):
        return self.width

    def get_grid_height(self):
        return self.height
    
    def __setitem__(self, (x, y), value):
        self.grid[x][y] = value

    def __getitem__(self, (x, y)):
        return self.grid[x][y]

    # adapts the grids representation of columns so that the layout is correct
    def column(self, col):
        return self.height - 1 - col

    # assign a value to each position on the grid
    def set_grid_values(self, coordinates, grid_value):
        for coord in coordinates:
            column_index = int(self.column(coord[1]))
            row_index = int(coord[0])
            self.grid[column_index][row_index] = grid_value