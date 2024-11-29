import random
from operators import *
from pyfiglet import figlet_format # Temp. for error messages

def check_interval(interval):
    """Return string from tuple"""
    return f"[{interval[0]}, {interval[1]}]"

def form2str(par, h):
    """Convert formula to string"""
    if par:
        return f"({form2str(False, h)})"

    # Predicate case
    if isinstance(h, Pred):
        return h.__2str__()

    elif isinstance(h, Var):
        return h.__str__()

    elif isinstance(h, str):
        return h

    # Equal case
    elif isinstance(h, Equal):
        return f"({form2str(False, h.var1)} = {form2str(False, h.var2)})"

    # Less case
    elif isinstance(h, Less):
        return f"({form2str(False, h.var1)} < {form2str(False, h.var2)})"

    # LessEq case
    elif isinstance(h, LessEq):
        return f"({form2str(False, h.var1)} <= {form2str(False, h.var2)})"

    # Negation case
    elif isinstance(h, Neg):
        return f"NOT {form2str(True, h.operator)}"

    # Exists case
    elif isinstance(h, Exists):
        return f"(EXISTS {h.var.__str__()}. {form2str(True, h.operator)})"

    # ForAll case
    elif isinstance(h, ForAll):
        return f"FORALL {h.var__str__()}. {form2str(True, h.operator)}"

    # Prev case
    elif isinstance(h, Prev):
        return f"PREVIOUS{check_interval(h.interval)} {form2str(True, h.operator)}"

    # Once case
    elif isinstance(h, Once):
        return f"ONCE{check_interval(h.interval)} {form2str(True, h.operator)}"

    # And case
    elif isinstance(h, And):
        return f"{form2str(False, h.operator1)} AND {form2str(True, h.operator2)}"

    # Or case
    elif isinstance(h, Or):
        return f"({form2str(False, h.operator1)} OR {form2str(True, h.operator2)})"

    # Implies case
    elif isinstance(h, Implies):
        return f"{form2str(False, h.operator1)} IMPLIES {form2str(True, h.operator2)}"

    # Since case
    elif isinstance(h, Since):
        return f"{form2str(False, h.operator1)} SINCE{check_interval(h.interval)} {form2str(True, h.operator2)}"

    # Until case
    elif isinstance(h, Until):
        return f"{form2str(False, h.operator1)} UNTIL{check_interval(h.interval)} {form2str(True, h.operator2)}"

    elif isinstance(h, Aggreg):
        if isinstance(h.group_vars, Var):
            group_vars_str = h.group_vars.name
        else:
            group_vars_str = ", ".join([v.name for v in h.group_vars])
        return f"({h.var.name} <- {h.operator} {h.var.name}; {group_vars_str} {form2str(True, h.formula)})"

    # Error case
    else:
        raise ValueError(f"Unsupported formula type: {type(h).__name__, h}")

class FormulaGenerator:
    def __init__(self, sig, size, seed, ub_fv=3, weights=None):
        self.sig = sig
        self.size = size
        self.max_arity = sig.max_arity
        self.min_arity = sig.min_arity
        self.upper_bound_fv = sig.max_arity #ub_fv
        self.test_new_var = set([Var(f"x{i}") for i in range(1, self.upper_bound_fv+1)])
        self.y_counter = 0
        self.rng = random.Random()
        if seed is None:
            self.rng.seed()
        else:
            self.rng.seed(seed)

        # todo: fix solution for this
        if self.upper_bound_fv < self.min_arity:
            print("__________\nWarning: upper bound of free variables is less than the minimum arity of predicates.")
            print(f"Temporary fix: upping the upper bound of free variables to {self.min_arity}.\n__________")
            self.upper_bound_fv = self.min_arity
            self.test_new_var = set([Var(f"x{i}") for i in range(1, self.upper_bound_fv+1)])

        if weights is None:
            self.weights = {
                'And': 0.1, 
                'Or': 0.1,
                'Prev': 0.1, 
                'Once': 0.1, 
                'Since': 0.1, 
                'Until': 0.1,
                'Rand': 0.1, 
                'Eand': 0.1, 
                'Nand': 0.1,
                'Exists': 0.1,
                'Aggreg': 0.1
            }
        else:
            self.weights = weights


    ### Old random_var function
    # def random_var(self, n=1, fv=set(), ub=None):
    #     """Return n variables."""
    #     if {x for x in fv if x.name.startswith('x')} == set():
    #         fv = fv.union(self.test_new_var)
    #     if ub is not None:
    #         if len(fv) < ub:
    #             new_var = self.test_new_var - fv
    #             if new_var != set():
    #                 fv = fv.union(self.rng.sample(new_var, k=ub-len(fv)))
    #     fv = sorted(fv, key = lambda x: x.name)
    #     if len(fv) < n:
    #         frs = self.rng.sample(fv, k=len(fv))
    #         snd = []
    #         for _ in range(n-len(fv)):
    #             snd.append(self.rng.sample(fv, k=1)[0])
    #         a = frs + snd
    #     else:
    #         a = self.rng.sample(fv, k=n)
    #     return a



    #### Old random_pred function
    # def random_pred(self, lb=0, ub=None, free_vars=None):
    #     """Return a Pred instance with variables."""
    #     if ub is None:
    #         #ub = len(free_vars.union(self.test_new_var))
    #         pred = self.rng.choice([p for p in self.sig.predicates if p.len >= lb])
    #     else:
    #         try:
    #             pred = self.rng.choice([p for p in self.sig.predicates if p.len >= lb and p.len <= ub])
    #         except IndexError:
    #             print("No predicates found with the given arity bounds.")
    #             print(f"Predicate bounds: {lb} <= arity <= {ub}")
    #             print("Available predicates:")
    #             for p in self.sig.predicates:
    #                 print(f"{p.name}({p.len})")
    #             pred = self.rng.choice([p for p in self.sig.predicates if p.len >= lb])
    #     if free_vars is None:
    #         vars_in_pred = self.random_var(pred.len, self.test_new_var, ub)
    #     else:
    #         vars_in_pred = self.random_var(fv = free_vars, n=pred.len, ub=ub)
    #     return Pred(pred.name, vars_in_pred)


    def random_var(self, n=1, fv=set(), lb=None, ub=None):
        """
        Return n variables from a given free variable set.
        Ensures the number of variables respects the lower and upper bounds.
        """
        if {x for x in fv if x.name.startswith('x')} == set():
            # Add new variables if no "x"-type variables exist
            fv = fv.union(self.test_new_var)

        if lb is not None and len(fv) < len(lb):
            # Introduce new variables to meet the lower bound
            missing = lb - len(fv)
            new_vars = list(self.test_new_var - fv)
            if len(new_vars) < missing:
                raise ValueError(f"Cannot meet lower bound of {lb}. Only {len(new_vars)} new variables available.")
            fv = fv.union(self.rng.sample(new_vars, k=missing))

        if ub is not None and len(fv) > len(ub):
            # Trim the set to meet the upper bound
            fv = set(sorted(fv, key=lambda x: x.name)[:len(ub)])

        if len(fv) < n:
            # Ensure at least n variables are selected
            frs = self.rng.sample(fv, k=len(fv))
            snd = [self.rng.choice(list(fv)) for _ in range(n - len(fv))]
            return frs + snd
        else:
            return self.rng.sample(fv, k=n)
    
    def random_pred(self, lb=set(), ub=None, free_vars=None, fv_lb=0):
        """
        Return a Pred instance with variables from free_vars.
        Respects the bounds on free variables.
        """
        if ub is None:
            ub = len(free_vars.union(self.test_new_var)) if free_vars else len(self.test_new_var)
        # Select a predicate within the bounds
        try:
            pred = self.rng.choice([p for p in self.sig.predicates if p.len >= max(len(lb), fv_lb) and p.len <= len(ub)])
        except IndexError:
            print(self.sig.__str__())
            print(f"No predicates found with the given arity bounds: {len(lb)} <= arity <= {len(ub)}")
            pred = self.rng.choice([p for p in self.sig.predicates if p.len >= max(len(lb), fv_lb)])
            # raise ValueError("No predicates found with the given arity bounds.")

        if free_vars is None:
            vars_in_pred = self.random_var(pred.len, fv=self.test_new_var, lb=lb, ub=ub)
        else:
            vars_in_pred = self.random_var(n=pred.len, fv=free_vars, lb=lb, ub=ub)

        return Pred(pred.name, vars_in_pred)

    def random_const(self):
        """Random constant."""
        return str(self.rng.randint(0, 100))

    # todo: add bounds for the interval and allow inf
    def random_interval(self, left_lb=0, left_ub=5, ub=20):
        """Random interval."""
        start = self.rng.randint(left_lb, left_ub)
        end = self.rng.randint(start+1, ub)  # end >= start
        return (start, end)

    def generate(self, free_vars = None, size=None, n_lb = 0, fv_lb = None, fv_ub = None):
        """Generate a random formula."""
        if size is None:
            size = self.size
        if free_vars is None:
            free_vars = set()
        if fv_lb is None:
            fv_lb = set()
        if fv_ub is None:
            fv_ub = self.test_new_var.union(free_vars).union(fv_lb)

        if size == 0:
            formula = self.random_pred(free_vars=free_vars, fv_lb = n_lb, lb=fv_lb.copy(), ub=fv_ub.copy())
            return formula, set(formula.variables)
        else:
            formula_choice = self.rng.choices(list(self.weights.keys()), weights=list(self.weights.values()))[0]

            if formula_choice == 'And':
                new_size = (size - 1) // 2
                subformula1, fv1 = self.generate(free_vars, new_size, fv_lb=fv_lb.copy(), fv_ub=fv_ub.copy(), n_lb=n_lb)
                subformula2, fv2 = self.generate(free_vars, new_size, fv_lb=fv_lb.copy(), fv_ub=fv_ub.copy(), n_lb=n_lb)
                formula = And(subformula1, subformula2)
                return formula, fv1.union(fv2)

            elif formula_choice == 'Rand':
                formula, fv = self.generate_and_with_relation(free_vars, size, fv_lb.copy(), fv_ub.copy())
                return formula, fv

            elif formula_choice == 'Eand':
                formula, fv = self.generate_and_with_equality(free_vars, size, fv_lb.copy(), fv_ub.copy()) # n_lb ???
                return formula, fv

            elif formula_choice == 'Nand':
                new_size = (size - 1) // 2
                subformula1, fv1 = self.generate(free_vars, new_size, fv_lb=fv_lb.copy(), fv_ub=fv_ub.copy(), n_lb=n_lb)
                subformula2, fv2 = self.generate(fv1.copy(), new_size, fv_lb=fv_lb.copy(),fv_ub=fv1.copy(), n_lb=n_lb)
                if not fv2.issubset(fv1):
                    new_fv = fv2 - fv1
                    self.Error_print(subformula1, new_fv, fv1.copy())
                formula = And(subformula1, Neg(subformula2))
                return formula, fv1.union(fv2)

            # elif formula_choice == 'Neg':
            #     # Rule: Variables in subformula must be empty
            #     subformula, fv_sub = generate(size - 1, sig, set())
            #     formula = Neg(subformula)
            #     return formula, fv_sub  # fv_sub should be empty

            elif formula_choice == 'Or':
                # Rule: fv(alpha) == fv(beta)
                new_size = (size - 1) // 2
                subformula1, fv1 = self.generate(free_vars, new_size, fv_lb=fv_lb.copy(), fv_ub=fv_ub.copy(), n_lb=n_lb)
                # print("Or: Free vars:", len(fv1),form2str(True, subformula1), [v.name for v in fv1], fv_lb, fv_ub)
                subformula2, fv2 = self.generate(fv1.copy(), new_size, fv_lb=fv1.copy(), fv_ub=fv1.copy(), n_lb=n_lb)
                if fv1!=fv2:
                    print(f"Or:\nform1: {form2str(True, subformula1)}\nform2: {form2str(True, subformula2)}\nfv1: {[v.name for v in fv1]} != fv2{[v.name for v in fv2]}")
                    if fv2-fv1 != set(): # if fv1 is missing some variables from fv2
                        missing_fv = fv2-fv1
                        self.Error_print(subformula1, missing_fv, fv1.copy())
                    if fv1-fv2 != set(): # if fv2 is missing some variables from fv1
                        missing_fv = fv1-fv2
                        self.Error_print(subformula2, missing_fv, fv2.copy())
                formula = Or(subformula1, subformula2)
                return formula, fv1

            elif formula_choice in ['Since', 'Until']:
                new_size = (size - 1) // 2
                interval = self.random_interval()
                # Rule: fv(alpha) ⊆ fv(beta)
                # Generate beta first
                subformula_beta, fv_beta = self.generate(free_vars, new_size, fv_lb=fv_lb.copy(), fv_ub=fv_ub.copy(), n_lb=n_lb)
                # print("Since/Until: Free vars:",interval, form2str(True, subformula_beta), len(fv_beta), [v.name for v in fv_beta], [v.name for v in fv_lb], [v.name for v in fv_ub])
                # Generate alpha with variables subset of fv_beta
                subformula_alpha, fv_alpha = self.generate(fv_beta.copy(), new_size, fv_lb=fv_lb.copy(), fv_ub=fv_beta.copy(), n_lb = n_lb) # fv_lb=min(fv_lb, len(fv_beta))
                if not fv_alpha.issubset(fv_beta):
                    print(f"Since/Until:\nBeta: {form2str(True, subformula_beta)}\nAlpha: {form2str(True, subformula_alpha)}\nfv_alpha: {[v.name for v in fv_alpha]} is not subset of fv_beta: {[v.name for v in fv_beta]}")
                    new_free = fv_alpha-fv_beta
                    self.Error_print(subformula_beta, new_free, fv_beta)
                formula_class = Since if formula_choice == 'Since' else Until
                formula = formula_class(interval, subformula_alpha, subformula_beta)
                return formula, fv_beta

            elif formula_choice == 'Prev':
                interval = self.random_interval()
                subformula, fv_sub = self.generate(free_vars, size - 1, fv_lb=fv_lb.copy(), fv_ub=fv_ub.copy(), n_lb=n_lb)
                formula = Prev(interval, subformula)
                return formula, fv_sub

            elif formula_choice == 'Once':
                interval = self.random_interval()
                subformula, fv_sub = self.generate(free_vars, size - 1, fv_lb=fv_lb.copy(), fv_ub=fv_ub.copy(), n_lb=n_lb)
                formula = Once(interval, subformula)
                return formula, fv_sub

            elif formula_choice == 'Exists':
                self.y_counter += 1
                var = Var(f"y{self.y_counter}")
                fv = free_vars.copy()
                fv.add(var)
                subformula, fv_sub = self.generate(fv, size - 1, fv_lb=fv_lb.copy(), fv_ub=fv_ub.copy(), n_lb=n_lb)
                formula = Exists(var, subformula)
                fv_sub.discard(var)
                return formula, fv_sub

            elif formula_choice == 'Aggreg':
                return self.generate_aggregation(free_vars, size, fv_lb.copy(), fv_ub.copy())

            else:
                raise ValueError(f"Unknown formula type chosen: {formula_choice}")

    def Error_print(self, f, fv, spare_vars=[]):
        """Fix the free variables with an And formula."""
        print(figlet_format('Andfix Error', font='big'))
        print(f"Fixing And2: {form2str(False, f)}, {[v.name for v in fv]}")


    def generate_and_with_relation(self, free_vars, size, lb, ub):
        """
        Generate a formula of the form alpha ∧ t1 R t2
        """
        new_size = size - 1
        if ub == set():
            subformula1, fv1 = self.generate(free_vars, new_size, fv_lb=lb, fv_ub=ub)
            subformula2, fv2 = self.generate(free_vars, new_size, fv_lb=lb, fv_ub=ub)
            formula = And(subformula1, subformula2)
            return formula, fv1.union(fv2)
        # Generate alpha
        alpha_formula, fv_alpha = self.generate(free_vars, new_size, fv_lb = lb, fv_ub=ub, n_lb=1) #fv_lb=max(1,lb),
        # Generate t1 R t2 with variables from fv_alpha or as a constant
        if len(fv_alpha) < 2:
            t1 = self.random_var(fv = fv_alpha)[0]
            t2 = self.random_const()
        else:
            t1, t2 = self.random_var(2, fv_alpha)
        term_set = {x for x in [t1, t2] if x.__class__.__name__ == 'Var'}
        if not term_set.issubset(fv_alpha):
            print(f"RAnd {form2str(True, alpha_formula)}:\nTerm set: {[v.name for v in term_set]} is not subset of fv_alpha: {[v.name for v in fv_alpha]}")
            self.Error_print(alpha_formula, term_set, fv_alpha)
        relation = self.rng.choice(['Equal', 'Less', 'LessEq'])
        if relation == 'Equal':
            relation_formula = Equal(t1, t2)
        elif relation == 'Less':
            relation_formula = Less(t1, t2)
        else:
            relation_formula = LessEq(t1, t2)
        formula = And(alpha_formula, relation_formula)
        return formula, fv_alpha

    def generate_and_with_equality(self, free_vars, size, lb, ub):
        """
        Generate a formula of the form alpha ∧ x = t
        """
        new_size = size - 1
        if ub == set():
            subformula1, fv1 = self.generate(free_vars, new_size, fv_lb=lb, fv_ub=ub)
            subformula2, fv2 = self.generate(free_vars, new_size, fv_lb=lb, fv_ub=ub)
            formula = And(subformula1, subformula2)
            return formula, fv1.union(fv2)
        # Generate alpha
        alpha_formula, fv_alpha = self.generate(free_vars, new_size, fv_lb = lb, fv_ub=ub, n_lb=1) #, fv_lb=max(1,lb)
        t = self.random_var(fv = fv_alpha)[0]
        x = self.random_const()
        equality_formula = Equal(x, t)
        formula = And(alpha_formula, equality_formula)
        return formula, fv_alpha.union({t if t.__class__.__name__ == 'Var' else None})

    def generate_aggregation(self, free_vars, size, lb, ub):
        """
        Generate an aggregation formula of the form:
        Ω(var GROUP BY group_vars WHERE formula)
        """
        print("free_vars:", [v.name for v in free_vars])
        if not free_vars:
            free_vars = self.test_new_var.copy()
            print("New free vars:", [v.name for v in free_vars])

        agg_var = self.random_var(n=1, fv=free_vars)[0]  # Choose the aggregation variable
        group_vars = set(self.random_var(n=max(lb, 1), fv=free_vars-{agg_var}))  # Randomly choose group vars
        print("Group vars:", [v.name for v in group_vars])
        print("Agg var:", agg_var.name)
        needed_vars = group_vars.union({agg_var})
        print("Needed vars:", [v.name for v in needed_vars])
        subformula, fv_sub = self.generate(needed_vars, size - 1, fv_lb=2, fv_ub=ub)#, fv_ub=len(needed_vars))
        print("test")

        agg_operator = self.rng.choice(['CNT', 'SUM', 'AVG', 'MIN', 'MAX']) # 'MED'

        formula = Aggreg(agg_operator, agg_var, group_vars, subformula)
        print("Aggregation formula:", form2str(False, formula))
        print("Free vars:", [v.name for v in fv_sub.union(set(group_vars))])
        return formula, needed_vars #group_vars#
