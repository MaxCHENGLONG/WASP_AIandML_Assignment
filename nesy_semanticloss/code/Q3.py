from pysdd.sdd import SddManager, Vtree # import pysdd.sdd as sdd

def build_exactly_one_constraint(n_vars = 10): # MNIST 
    # Build a vtree for n_vars variables
    vtree = Vtree(n_vars, var_order=list(range(1, n_vars + 1)), vtree_type="balanced")
    manager = SddManager.from_vtree(vtree)
    # create 10 variables
    variables = [manager.literal(i) for i in range(1, n_vars + 1)]
    # Build the exactly-one constraint: (x1 ∨ x2 ∨ ... ∨ xn) ∧ ¬(x1 ∧ x2) ∧ ... ∧ ¬(xn-1 ∧ xn)
    
    at_least_one = variables[0]
    for var in variables[1:]:
        at_least_one = at_least_one | var

    at_most_one = manager.true()
    for i in range(n_vars):
        for j in range(i + 1, n_vars):
            not_both = (~variables[i]) | (~variables[j])
            at_most_one &= not_both
    
    exactly_one = at_most_one & at_least_one  # at least one is true

    print("Exactly-One SDD size:", exactly_one.size())
    print("Exactly-One SDD model count:", exactly_one.model_count())
    return manager, exactly_one

manager, constraint = build_exactly_one_constraint(n_vars=10)


