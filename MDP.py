import numpy as np
import matplotlib.pyplot as plt

# Gamma
discount = 0.8
actions = [[-1,0], [1,0], [0,-1], [0,1]] # Up -> Down -> Left -> Right
num_actions = len(actions)
action_symbol = ['^','v','<','>']


class MDP:

    def __init__(self, states, rewards):
        self.states = states
        self.rewards = rewards
        self.p = 0.8

        len_row = len(states[0])
        len_col = len(states)

        self.tr_model = np.empty((len_col,len_row,num_actions,len_col,len_row))

    def init_MDP(self):
        len_row = len(self.states[0])
        len_col = len(self.states)

        # For each state (i,j) and (m,n), calculate the proabaility to arrive at (m,n) if action a
        # is executed from (i,j)
        for i in range(len_col):
            for j in range(len_row):
                for a in range(num_actions):
                    for m in range(len_col):
                        for n in range(len_row):
                            prob = self.calc_prob(self.states, i,j,a,m,n)
                            self.tr_model[i][j][a][m][n] = prob
                            


    def calc_prob(self,states,i,j,a,m,n):

        curr = states[i][j]
        next = states[m][n]

        # vertical - a tells us which action - 0 is vertical, 1 is horizontal
        v = actions[a][0]
        # horizontal 
        h = actions[a][1] 

        # s is a blocked or terminal state, or s' is blocked
        if(curr == '0' or curr == -1 or next == '0'):
            return 0
        # state out of reach
        elif(abs(i-m)>1 or abs(j-n)>1): 
            return 0

        # If (m,n) is diagonal to (i,j) in the direction of a
        if(v != 0):

            # The probability to transfer to either diagonal of (i,j) in the direction of action a
            # is (1-p)/2
            if((m == i + v and n == j + v) or (m == i + v and n == j - v)):
                return (1-self.p)/2
        elif(h != 0):
            if((m == i + h and n == j + h) or (m == i - h and n == j + v)):
                return (1-self.p)/2

        # If (m,n) is the intended result of action a from (i,j) 
        if(i+v==m and j+h==n):
            # Initially probability = p
            prob = self.p

            # If the action is vertical (up or down)
            if(v != 0):
                # Check if diagonal to (i,j) in the direction of action a is blocked. If it is, add
                # (1-p)/2 to probaility of action success
                if(0 <= j+v < len(self.states)):
                    if(self.states[i+v][j+v]==0):
                        prob = prob + (1-self.p)/2
                if(0 <= j-v < len(self.states)):
                    if(self.states[i+v][j-v]==0):
                        prob = prob + (1-self.p)/2

            # If the action is horizontal (left or right)
            elif(h != 0):
                 # Check if diagonal to (i,j) in the direction of action a is blocked. If it is, add
                # (1-p)/2 to probaility of action success
                if(0 <= i+h < len(self.states)):
                    if(self.states[i+h][j+h]==0):
                        prob = prob + (1-self.p)/2
                if(0 <= i-h < len(self.states)):
                    if(self.states[i-h][j+h]==0):
                        prob = prob + (1-self.p)/2
            return prob

        return 0


def main():

    # Read input
    filename = "input1_2026a.npz"

    data = np.load(filename)

    states = data['states']

    rewards = data['rewards']

    # Initiallize MDP
    mdp = MDP(states, rewards)

    mdp.init_MDP()

    # Run value iteration (in this case, epsilon = 0.0001)
    val_it = value_iteration(mdp, 0.0001)

    # Utilility matrix that is outputted from value iteration
    util_matrix = val_it[0]

    # Number of iterations requred by value iteration to converge
    num_iterations = val_it[2]

    # Policy found by value iteration
    policy = val_it[1]

    # Present the utility matrix from value iteration as a ceismic map
    maxAbsReward = np.max(abs(mdp.rewards))

    fig = plt.imshow(util_matrix, vmin=-maxAbsReward, vmax=maxAbsReward, cmap='seismic')

    plt.title(f'Ido Levy - Value Iteration: {num_iterations}')

    plt.colorbar()

    # fname = ""
    # plt.savefig('' + fname + '_ValueIteration_Values.jpg')

    plt.show()

    # Print the policy found by value iteration
    print("\nPolicy found by value iteration:")
    for i in range(len(policy)):
        for j in range(len(policy[i])):
            print(policy[i][j], end="  ")
        print()


    # Run policy iteration
    pol_it = policy_iteration(mdp)

    # Policy matrix found by policy iteration
    pol = pol_it[0]

    # Utility matrix found by polcy iteration
    ut_matrix = pol_it[1]

    # n_iters[i] = number of iterations done by simplified value iteration 
    # during the ith iteration of policy iteration
    n_iters = pol_it[2]

    # Plot iterations graph
    y_points = np.array(pol_it[3])
    x_points = np.array(list(range(len(y_points))))

    plt.plot(x_points, y_points)
    plt.plot(x_points, y_points, marker='o')
    plt.xlabel('Policy-iteration iteration num')
    plt.ylabel('Number of policy-evaluation iterations')
    plt.title('policy iteration vs evaluation iterations')
    # fname = ""
    # plt.savefig('' + fname + '_PolicyIteration_IdoLevy.jpg')
    plt.show()

    # Present the utility matrix from value iteration as a ceismic map
    maxAbsReward = np.max(abs(mdp.rewards))

    fig = plt.imshow(ut_matrix, vmin=-maxAbsReward, vmax=maxAbsReward, cmap='seismic')

    plt.title(f'Ido Levy - Policy Iteration: {n_iters}')

    plt.colorbar()

    # fname = ""
    # plt.savefig('' + fname + '_ValueIteration_Values.jpg')

    plt.show()


    # Print the policy found by policy iteration
    print("\nPolicy found by policy iteration:")
    for i in range(len(states)):
        for j in range(len(states[i])):
            if(mdp.states[i][j] == 1):
                print(action_symbol[int(pol[i][j])], end="  ")
            elif(mdp.states[i][j] == 0):
                print("x", end = "  ")
            else:
                print("o", end = "  ")
        print()



def value_iteration(mdp, e):

    # U_i+1 vextor
    utils = np.zeros((len(mdp.states),len(mdp.states[0])))

    # U_i vector
    prev_util = np.zeros((len(mdp.states),len(mdp.states[0])))

    # Policy found by value iteration
    policy = np.full((len(mdp.states),len(mdp.states[0])),'x',dtype='U1')

    num_iterations = 0

    # Until Convegence
    while True:
        # U_i
        prev_util = utils.copy()

        # Max error
        delta = 0

        # Go over all states s
        for i in range(len(mdp.states)):
            for j in range(len(mdp.states[0])):

                # If (i,j) is blocked or terminal, continue
                if(mdp.states[i][j] == -1):
                    utils[i][j] = mdp.rewards[i][j]
                    policy[i][j] = 'o'
                    continue
                elif(mdp.states[i][j] == 0):
                    utils[i][j] = 0
                    policy[i][j] = 'x'
                    continue

                maxUtil = -float('inf')
                action = -1

                # For all actions
                for a in range(num_actions):

                    sum_u = q_value(mdp,i,j,a,prev_util)

                    # IF SUM_U == sum_u across actions with max utils, then place a + (indifferent to best action)
                    if(sum_u == maxUtil):
                        action = '+'
                    elif(sum_u > maxUtil):
                        maxUtil = sum_u
                        action = a

                    
                utils[i][j] = maxUtil

                # delta = ||U_i+1 - U_i||
                if(abs(maxUtil - prev_util[i][j]) > delta):
                    delta = maxUtil - prev_util[i][j]

                if(action != -1 and action != '+'):
                    policy[i][j] = action_symbol[action]
                elif(action == '+'):
                    policy[i][j] = '+'

        num_iterations = num_iterations+1

        # Check for convergance
        if(discount != 0 and delta <= (e*(1-discount))/discount):            
            return [utils,policy,num_iterations]
        elif(delta <= e):
            return [utils,policy,num_iterations]


        
def reachable_states(states, i, j, a):

    # Reaxhable states from (i,j) given action a
    reachable = []

    # vertical - a tells us which action, 0 is in the vertical direction, 1 is horizontal direction
    v = actions[a][0]
    # horizontal 
    h = actions[a][1]
     
    # Check diagonals from (i,j) in direction of a
    if(0 <= i + v < len(states) and 0 <= j + h < len(states[0])):
        reachable.append([i+v,j+h])
        if(v != 0):
            if(0 <= j + v < len(states[0])):
                reachable.append([i+v,j+v])
            if(0 <= j - v < len(states[0])):
                reachable.append([i+v,j-v])
        elif(h != 0):
            if(0 <= i + h < len(states)):
                reachable.append([i+h,j+h])
            if(0 <= i - h < len(states[0])):
                reachable.append([i-h,j+h])

    return reachable

    

def q_value(mdp, i,j, a, prev_util):

    # Action a
    a = int(a)

    reachable = reachable_states(mdp.states,i,j,a)

    sum_u = 0

    # For all states s' reachable form s
    for sTag in reachable:
        p_sTag_given_sa = mdp.tr_model[i][j][a][sTag[0]][sTag[1]] # P(s'|s,a)
        reward = mdp.rewards[sTag[0]][sTag[1]]

        sum_u = sum_u + p_sTag_given_sa * (reward + discount*prev_util[sTag[0]][sTag[1]])
            
    return sum_u



def policy_iteration(mdp):

    utils = np.zeros((len(mdp.states),len(mdp.states[0])))
    policy = np.zeros((len(mdp.states),len(mdp.states[0])))
    num_iterations = 0

    # Determinse convergence codition for policy evaluation
    epsilon = 0.001

    # n_iters[i] = number of iterations done by simplified value iteration 
    # during the ith iteration of policy iteration
    n_iterations_policy_eval = []

    # initialize terminal utils with corssesponding reward
    for i in range (len(mdp.states)):
        for j in range(len(mdp.states[0])):

            if(mdp.states[i][j] == -1):
                utils[i][j] = mdp.rewards[i][j]
                policy[i][j] = -1
            elif(i == 0):
                policy[i][j] = 1

    while True:

        unchanged = True
        num_iterations = num_iterations + 1
        
        # Perform policy evaluation
        policy_eval = policy_evaluation(mdp,policy,utils,epsilon)

        # Utilities evalueated for current policy
        utils = policy_eval[0]

        n_iterations_policy_eval.append(policy_eval[1])

        # For every state s
        for i in range(len(mdp.states)):
            for j in range(len(mdp.states[0])):

                if(mdp.states[i][j] != 1):
                    continue
                
                max_a = 0
                max_u = -float('inf')

                # Find action a that maximizes q_value from (i,j)
                for a in range(num_actions):
                    q_val = q_value(mdp,i,j,a,utils)

                    if(q_val > max_u):
                        max_u = q_val
                        max_a = a
                
                if(max_a != policy[i][j]):
                    policy[i][j] = max_a
                    unchanged = False
        
        # Policy has converged
        if(unchanged):
            return [policy,utils,num_iterations, n_iterations_policy_eval]
        
                    

# Implemented using simplified value iteration
def policy_evaluation(mdp, policy, utils, epsilon):

    u = utils.copy()
    num_iterations = 0

    while True:

        prev_utils = u.copy()
        num_iterations = num_iterations + 1

        for i in range(len(mdp.states)):
            for j in range(len(mdp.states[0])):

                if(mdp.states[i][j]==0):
                    continue

                if(mdp.states[i][j]==-1):
                    u[i][j] = mdp.rewards[i][j]
                    continue
                
                action = policy[i][j]
                u[i][j] = q_value(mdp,i,j,action,prev_utils)

        max_diff = np.max(np.abs(u - prev_utils))
    
        # Check for convergence
        if(max_diff < epsilon):
                return [u,num_iterations]


main()