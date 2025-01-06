import numpy as np
from pulp import *
import time
"""
The user can directly use generate problem method to create a transportation problem and then use the 
helper functions to convert the result to LP and solve it by our revised simplex, or give them to solver
method.
The methods are used as an example at the end of this file , you can examine that part and directly run it with
different supply and node sizes.

There is a printing part in a comment in revised simplex method, you can use if you want to see the basic variables


"""

def get_variable(token):
    # Extracts variable name and its coefficient from the token
    # 1.2x0 -> x0, 1.2
    token = token.strip()
    x_index = token.index("x")
    variable = token[x_index:]
    coefficient = token[:x_index]
    if coefficient == "+" or coefficient == "":
        coefficient = 1
    elif coefficient == "-":
        coefficient = -1
    else:
        coefficient = float(coefficient)
    return variable, coefficient

class LP:
    def __init__(self):
        self.constraints = []
        self.cost = np.zeros(100000) # I am assuming number of variables are not more than 100000
        self.type = None
        self.non_negativity_constraints = None
        self.var_set = set()
        self.variables = []
        self.constraint_signs = []
        self.rhs = []
        self.coefficients = None
        self.artificial_var_indexes = []
        self.basis = []
        self.objective_value = 0

    def add(self, constraint):
        # Adding all variables to the set
        for token in constraint.split():
            if "x" in token:
                if token == "max":
                    # Max has x in it so don't parse it
                    continue
                variable, _ = get_variable(token)
                if variable not in self.var_set:
                    self.var_set.add(variable)
                    self.variables.append(variable)
        if "min" in constraint or "max" in constraint:
            self.type = constraint.split()[0]
            constraint = constraint.split(" ")[1:]
            for token in constraint:
                if "x" in token:
                    variable, coefficient = get_variable(token)
                    column_index = self.variables.index(variable)
                    self.cost[column_index] = coefficient
        elif "," in constraint:
            # I am assuming non-negativity constraints come last
            # So I am reducing the size of the cost vector
            self.non_negativity_constraints = constraint
            self.cost = self.cost[: len(self.variables)]
        else:
            self.constraints.append(constraint)
            if ">=" in constraint:
                self.constraint_signs.append(">=")
            elif "<=" in constraint:
                self.constraint_signs.append("<=")
            else:
                self.constraint_signs.append("=")
            self.rhs.append(float(constraint.split("=")[-1]))

    def get_coefficient_matrix(self):
        # Returns the coefficient matrix of the constraints
        # Columns are variables and rows are constraints
        matrix = np.zeros((len(self.constraints), len(self.variables)))
        for index, constraint in enumerate(self.constraints):
            for token in constraint.split():
                if "x" in token:
                    variable, coefficient = get_variable(token)
                    column_index = self.variables.index(variable)
                    matrix[index][column_index] = coefficient
        self.coefficients = matrix

    def add_surplus_variable(self, coeff_index, sign=1):
        # Adds a new variable with cost 0
        variable_name = f"s{len(self.variables) + 1}"
        self.variables.append(variable_name)
        self.cost = np.append(self.cost, 0)
        slack = np.zeros(len(self.constraints))
        slack[coeff_index] = sign
        self.coefficients = np.column_stack((self.coefficients, slack))
        if sign == 1:
            self.basis.append(len(self.variables) - 1)

    def add_artificial_variable(self, coeff_index):
        # Adds a new variable with cost -M or M (changes with problem type)
        variable_name = f"a{len(self.variables) + 1}"
        self.variables.append(variable_name)
        base_M = np.max(np.abs(self.cost)) + 1
        M = -base_M
        if self.type == "min":
            M = base_M
        self.cost = np.append(self.cost, M)
        self.artificial_var_indexes.append(len(self.variables) - 1)
        artificial = np.zeros(len(self.constraints))
        artificial[coeff_index] = 1
        self.coefficients = np.column_stack((self.coefficients, artificial))
        self.basis.append(len(self.variables) - 1)

    def convert_non_negativity_constraints(self):
        for non_negativity_constraint in self.non_negativity_constraints.split(","):
            # If x0 <= 0 then we can replace x0 with x0' = x0
            if "<=" in non_negativity_constraint:
                variable, _ = get_variable(non_negativity_constraint.split("<=")[0])
                column_index = self.variables.index(variable)
                self.variables[column_index] = f"{variable}'"
                self.cost[column_index] *= -1
                self.coefficients[:,column_index] *= -1
            # If x0 free we change change it to x0 = x0_p - x0_n
            elif "free" in non_negativity_constraint:
                variable, _ = get_variable(non_negativity_constraint.split("free")[0])
                column_index = self.variables.index(variable)
                self.variables[column_index] = f"{variable}_p"
                self.variables.insert(column_index + 1, f"{variable}_n")
                self.cost = np.insert(
                    self.cost, column_index + 1, -self.cost[column_index]
                )
                self.coefficients = np.insert(
                    self.coefficients,
                    column_index + 1,
                    -self.coefficients[:, column_index],
                    axis=1,
                )

    def to_standard_form(self):
        # Adding slack and artificial variables
        for index, inequality in enumerate(self.constraint_signs):
            if inequality == "<=":
                self.add_surplus_variable(index, 1)
            elif inequality == ">=":
                self.add_surplus_variable(index, -1)
                self.add_artificial_variable(index)
            else:
                self.add_artificial_variable(index)
        if self.type == "min":
            self.cost = -self.cost
    
    def revised_simplex(self):
        # Initializing necessary matrices
        self.get_coefficient_matrix()
        self.convert_non_negativity_constraints()
        self.to_standard_form()
        A = self.coefficients.copy()

        solved = False
        self.variables = np.array(self.variables)
        var_indexes = np.arange(len(self.variables))
        basic_indexes = self.basis
        non_basic_indexes = np.delete(var_indexes, basic_indexes)
        B_inv = np.eye(len(basic_indexes))
        iteration = 0
        while not solved:
            
            c_b = self.cost[basic_indexes]
            c_n = self.cost[non_basic_indexes]
            N = A[:,non_basic_indexes]

            optimality_check = np.matmul(np.matmul(c_b, B_inv), N) - c_n
            entering_index_in_nonbasic = np.argmin(optimality_check)

            if optimality_check[entering_index_in_nonbasic] >= 0:
                solved = True
                break

            entering_index = var_indexes[non_basic_indexes][entering_index_in_nonbasic]
            
            new_rhs = np.matmul(B_inv, self.rhs)
            coeffs = np.matmul(B_inv,A[:,entering_index])
            # To avoid division by zero
            ratios = np.full(len(coeffs),float("inf"))
            ratios[coeffs > 0] = new_rhs[coeffs > 0] / coeffs[coeffs > 0]

            if np.all(ratios == float("inf")):
                print("Problem is unbounded")
                return float("inf")

            leaving_index_in_basic = np.argmin(ratios)
            leaving_index = var_indexes[basic_indexes][leaving_index_in_basic]

            # Updating B_inv
            E = np.eye(len(basic_indexes))
            E[:, leaving_index_in_basic] = -coeffs / coeffs[leaving_index_in_basic]
            E[leaving_index_in_basic, leaving_index_in_basic] = 1 / coeffs[leaving_index_in_basic]
            B_inv = np.matmul(E, B_inv)
            # Exchanging variables
            basic_indexes[leaving_index_in_basic] = entering_index
            non_basic_indexes[entering_index_in_nonbasic] = leaving_index
            iteration += 1
        
                
        result = np.matmul(np.matmul(c_b,B_inv),self.rhs)
        self.rhs = np.matmul(B_inv,self.rhs)
        # Non-zero artificial variable in the basis. Problem is infeasible
        for index in basic_indexes:
            if self.variables[index][0] == "a" and self.rhs[basic_indexes.index(index)] != 0:
                print("Problem is infeasible")
                return None
        # If minimization, we have the revert the result
        # Initialize an array for all variable values
        #all_variable_values = np.zeros(len(self.variables))
        # Assign the values to the corresponding basis variable indices
        #all_variable_values[basic_indexes] = self.rhs
        #print(f"All variable values: {all_variable_values}")
        if self.type == "min":
            result *= -1
        return result
    
def generate_problem(num_supply_nodes, num_demand_nodes, max_cost, max_demand_supply):
    # To avoid numpy random generator error where low == high
    if max_demand_supply == 1:
        supply_nodes = np.ones(num_supply_nodes,dtype=int)
        demand_nodes = np.ones(num_demand_nodes,dtype=int)

    else:
        supply_nodes = np.random.randint(1, max_demand_supply, num_supply_nodes)
        demand_nodes = np.random.randint(1, max_demand_supply, num_demand_nodes)

    # To avoid numpy random generator error where low == high
    if max_cost == 1:
        cost_matrix = np.ones((num_supply_nodes, num_demand_nodes),dtype=int)

    else:
        cost_matrix = np.random.randint(1, max_cost, (num_supply_nodes, num_demand_nodes))

    total_supply = np.sum(supply_nodes)
    total_demand = np.sum(demand_nodes)
    diff = abs(total_supply - total_demand)
    # We are trying to make supply and demand equal
    if total_supply > total_demand:
        index = 0
        while diff > 0 and index < len(demand_nodes):
            to_add = min(diff, max_demand_supply - demand_nodes[index])
            diff -= to_add
            demand_nodes[index] += to_add
            index += 1
    else:
        index = 0
        while diff > 0 and index < len(supply_nodes):
            to_add = min(diff, max_demand_supply - supply_nodes[index])
            diff -= to_add
            supply_nodes[index] += to_add
            index += 1

    return supply_nodes, demand_nodes, cost_matrix


def create_supply_dict(nodes):
    node_list=[]
    dict={}
    for i in range(len(nodes)):
        key = i
        node_list.append(key)
        dict[key] = nodes[i]
    return node_list,dict

def create_demand_dict(nodes):
    node_list=[]
    dict={}
    for i in range(len(nodes)):
        node_list.append(i)
        dict[i] = nodes[i]
    return node_list,dict

def solver(supply_nodes, demand_nodes, cost_matrix):
    supply_nodes_list, supply_dict = create_supply_dict(supply_nodes)
    demand_node_list, demand_dict = create_demand_dict(demand_nodes)
    prob = LpProblem("Beer Distribution Problem",LpMinimize)
    Routes = [(w,b) for w in supply_nodes_list for b in demand_node_list]
    route_vars = LpVariable.dicts("Route",(supply_nodes_list,demand_node_list),0,None)
    prob += lpSum([route_vars[w][b]*cost_matrix[supply_nodes_list.index(w)][b] for (w,b) in Routes]), "Sum of Transporting Costs"

    # The supply maximum constraints are added to prob for each supply node (warehouse)
    for w in supply_nodes_list:
        prob += lpSum([route_vars[w][b] for b in demand_node_list]) == supply_dict[w]

    # The demand minimum constraints are added to prob for each demand node (bar)
    for b in demand_node_list:
        prob += lpSum([route_vars[w][b] for w in supply_nodes_list]) == demand_dict[b]

    start_time = time.time()
    prob.solve()
    end_time = time.time()
    execution_time = end_time - start_time
    return execution_time, prob.objective.value()

def convert_to_LP(supply_nodes_list,demand_node_list,supply_dict,demand_dict,cost_matrix):
    lp_problem = LP()
    objective = "min "
    for i, supply in enumerate(supply_nodes_list):
        for j, demand in enumerate(demand_node_list):
            cost = cost_matrix[i][j]
            objective += f" +{cost}x{supply}_{demand}"
    lp_problem.add(objective.strip())
    # Add supply constraints to LP
    for supply in supply_nodes_list:
        constraint = ""
        for demand in demand_node_list:
            if demand==0:
                  constraint += f" x{supply}_{demand}"
            else:
                constraint += f" +x{supply}_{demand}"
        constraint += f" = {supply_dict[supply]}"
        lp_problem.add(constraint.strip())
    for demand in demand_node_list:
        constraint = ""
        for supply in supply_nodes_list:
            if supply_nodes_list.index(supply)==0:
                 constraint += f" x{supply}_{demand}"
            else:
                constraint += f" +x{supply}_{demand}"
        constraint += f" = {demand_dict[demand]}"
        lp_problem.add(constraint.strip())
    non_negativity_constraints = ""
    for supply in supply_nodes_list:
        for demand in demand_node_list:
            non_negativity_constraints += f"x{supply}_{demand} >= 0, "
    non_negativity_constraints = non_negativity_constraints[:-2]
    lp_problem.add(non_negativity_constraints)
    start_time= time.time()
    result = lp_problem.revised_simplex()
    end_time= time.time()
    execution_time= end_time- start_time
    print(f"Execution time of our revised simplex: {execution_time} seconds")
    return execution_time, result


num_runs = 10
total_revised_time = 0
total_solver_time = 0

for _ in range(num_runs):
    supply_nodes, demand_nodes, cost_matrix = generate_problem(10, 10, 20, 20)
    
    # Time for our revised simplex method
    supply_nodes_list, supply_dict = create_supply_dict(supply_nodes)
    demand_node_list, demand_dict = create_demand_dict(demand_nodes)
    revised_time, revised_result = convert_to_LP(supply_nodes_list, demand_node_list, supply_dict, demand_dict, cost_matrix)
    total_revised_time += revised_time
    
    # Time for the solver method
    solver_time, solver_result = solver(supply_nodes, demand_nodes, cost_matrix)
    total_solver_time += solver_time

    print(f"Our result : {revised_result}, PuLP result: {solver_result}")

# Calculating and printing the average execution times
avg_revised_time = total_revised_time / num_runs
avg_solver_time = total_solver_time / num_runs
print(f"Average execution time of our revised simplex: {avg_revised_time} seconds")
print(f"Average execution time of solver: {avg_solver_time} seconds")

"""
Example usage:
lp = LP()
lp.add("min 2x1 +x2")
lp.add("x1 +x2 >= 2")
lp.add("x1 -2x2 <= 4")
lp.add("x1 >= 0, x2 >= 0")
lp.revised_simplex()
Make sure that + or - is adjacent to variable to avoid errors
x1 +x2 is correct but x1 + x2 is not
"""
