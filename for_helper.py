import random
from operators import *

# def string_of_var(var):
#     """In case future var is not a string"""
#     return var.name

def check_interval(interval):
    """Return string from tuple"""
    return f"[{interval[0]}, {interval[1]}]"

def form2str(par, h):
    """Convert formula to string"""
    if par:
        return f"({form2str(False, h)})"

    # Predicate case
    if isinstance(h, Pred):
        # return string_of_predicate(h)
        return h.__2str__()

    elif isinstance(h, Var):
        # return string_of_var(h)
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

    # Error case
    else:
        raise ValueError(f"Unsupported formula type: {type(h).__name__, h}")

class FormulaGenerator:
    def __init__(self, sig, size, seed, ub_fv=3, weights=None):
        self.sig = sig
        self.size = size
        self.max_arity = sig.max_arity
        self.min_arity = sig.min_arity
        self.upper_bound_fv = 2 #sig.max_arity #ub_fv
        self.test_new_var = set([Var(f"x{i}") for i in range(1, self.upper_bound_fv+1)])
        self.y_counter = 0
        self.rng = random.Random()
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
                'Exists': 0.1
            }
        else:
            self.weights = weights 

    def random_var(self, n=1, fv=set()):
        """Return n variables."""
        if len(fv) < n:
            # Introduce new variables if needed
            fv = fv.union(self.test_new_var)
        fv = sorted(fv, key = lambda x: x.name)
        if len(fv) < n:
            frs = self.rng.sample(fv, k=len(fv))
            snd = self.rng.sample(self.test_new_var, k=n-len(fv))
            a = frs + snd
        else:
            a = self.rng.sample(fv, k=n)
        return a #self.rng.sample(fv, k=n)#test

    def random_pred(self, lb=0, ub=None, free_vars=None):
        """Return a Pred instance with variables."""
        if ub is None:
            ub = len(free_vars.union(self.test_new_var))
        pred = self.rng.choice([p for p in self.sig.predicates if p.len >= lb])# and p.len <= ub])
        if free_vars is None:# or free_vars == set():
            vars_in_pred = self.random_var(pred.len, self.test_new_var)
        else:
            vars_in_pred = self.random_var(fv = free_vars, n=pred.len) #.union(self.test_new_var)
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

    def generate(self, free_vars = None, size=None):
        """Generate a random formula."""
        if free_vars is None:
            free_vars = set()
        if size is None:
            size = self.size

        if size == 0:
            formula = self.random_pred(free_vars=free_vars)
            return formula, set(formula.variables)
        else:
            formula_choice = self.rng.choices(list(self.weights.keys()), weights=list(self.weights.values()))[0]

            if formula_choice == 'And':
                new_size = (size - 1) // 2
                subformula1, fv1 = self.generate(free_vars, new_size)
                subformula2, fv2 = self.generate(free_vars, new_size)
                formula = And(subformula1, subformula2)
                return formula, fv1.union(fv2)

            elif formula_choice == 'Rand':
                formula, fv = self.generate_and_with_relation(free_vars, size)
                return formula, fv

            elif formula_choice == 'Eand':
                formula, fv = self.generate_and_with_equality(free_vars, size)
                return formula, fv

            elif formula_choice == 'Nand':
                new_size = (size - 1) // 2
                subformula1, fv1 = self.generate(free_vars, new_size)
                subformula2, fv2 = self.generate(fv1.copy(), new_size)
                if not fv2.issubset(fv1):
                    new_fv = fv2 - fv1
                    subformula1, nfv = self.fix_And2(subformula1, new_fv, fv1.copy())
                    fv1 = fv1.union(nfv)
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
                subformula1, fv1 = self.generate(free_vars, new_size)
                subformula2, fv2 = self.generate(free_vars, new_size)
                if fv1!=fv2:
                    if fv2-fv1 != set(): # if fv1 is missing some variables from fv2
                        missing_fv = fv2-fv1
                        subformula1, new_fv = self.fix_And2(subformula1, missing_fv, fv1.copy())#And(subformula1, subformula2)
                        fv1 = fv1.union(new_fv)
                    if fv1-fv2 != set(): # if fv2 is missing some variables from fv1
                        missing_fv = fv1-fv2
                        subformula2, new_fv = self.fix_And2(subformula2, missing_fv, fv2.copy())#And(subformula2, subformula1)
                        fv2 = fv2.union(new_fv)
                    new_fv = fv1.union(fv2)
                    formula = Or(subformula1, subformula2)
                    return formula, new_fv
                formula = Or(subformula1, subformula2)
                return formula, fv1

            elif formula_choice in ['Since', 'Until']:
                new_size = (size - 1) // 2
                interval = self.random_interval()
                # Rule: fv(alpha) ⊆ fv(beta)
                # Generate beta first
                subformula_beta, fv_beta = self.generate(free_vars, new_size)
                # Generate alpha with variables subset of fv_beta
                subformula_alpha, fv_alpha = self.generate(fv_beta.copy(), new_size)
                if not fv_alpha.issubset(fv_beta):
                    new_free = fv_alpha-fv_beta
                    subformula_beta, nf = self.fix_And2(subformula_beta, new_free, fv_beta)#And(self.fix_And(new_free), subformula_alpha)
                    fv_beta = fv_beta.union(nf)
                formula_class = Since if formula_choice == 'Since' else Until
                formula = formula_class(interval, subformula_alpha, subformula_beta)
                return formula, fv_beta

            elif formula_choice == 'Prev':
                interval = self.random_interval()
                subformula, fv_sub = self.generate(free_vars, size - 1)
                formula = Prev(interval, subformula)
                return formula, fv_sub

            elif formula_choice == 'Once':
                interval = self.random_interval()
                subformula, fv_sub = self.generate(free_vars, size - 1)
                formula = Once(interval, subformula)
                return formula, fv_sub
            
            elif formula_choice == 'Exists':
                self.y_counter += 1
                var = Var(f"y{self.y_counter}")
                fv = free_vars.copy()
                fv.add(var)
                subformula, fv_sub = self.generate(fv, size - 1)
                formula = Exists(var, subformula)
                fv_sub.discard(var)
                return formula, fv_sub

            else:
                raise ValueError(f"Unknown formula type chosen: {formula_choice}")
    
    def fix_And2(self, f, fv, spare_vars=[]):
        """Fix the free variables with an And formula."""
        print(f"Fixing And2: {form2str(False, f)}, {[v.name for v in fv]}")
        preds = []
        new_fv = set()
        if [p for p in self.sig.predicates if p.len >= len(fv) and p.len <= len(fv)+len(spare_vars)]!=[]:
            pred = self.rng.choice([p for p in self.sig.predicates if p.len >= len(fv) and p.len <= len(fv)+len(spare_vars)])
            pred_vars = self.rng.sample(fv, k=len(fv))
            if pred.len > len(fv): 
                pred_vars.extend(self.random_var(pred.len - len(fv), spare_vars)) # add new variables from the spare_vars
            pred = Pred(pred.name, pred_vars)
            new_fv = new_fv.union(pred.variables)
            out_form = And(pred, f)
        elif len(fv) > self.max_arity:
            pred = self.random_pred(lb=1, free_vars=fv)
            preds.append(pred)
            new_fv = new_fv.union(pred.variables)
            out_form = And(pred, f)
            return self.fix_And2(out_form, (fv - new_fv), spare_vars)
        else:
            pred = self.random_pred(lb=1, free_vars=fv)
            new_fv = new_fv.union(pred.variables)
            out_form = And(pred, f)
            return self.fix_And2(out_form, (fv - new_fv))
        print(f"Returning fixed formula: {form2str(False, out_form)}")
        return out_form, fv.union(new_fv)


    def generate_and_with_relation(self, free_vars, size):
        """
        Generate a formula of the form alpha ∧ t1 R t2
        """
        new_size = size - 1
        # Generate alpha
        alpha_formula, fv_alpha = self.generate(free_vars, new_size)
        # Generate t1 R t2 with variables from fv_alpha or as a constant
        if len(fv_alpha) < 2:
            t1 = self.random_var(fv = fv_alpha)[0]
            t2 = self.random_const()
        else:
            t1, t2 = self.random_var(2, fv_alpha)
        term_set = {x for x in [t1, t2] if x.__class__.__name__ == 'Var'}
        if not term_set.issubset(fv_alpha):
            alpha_formula, fv = self.fix_And2(alpha_formula, term_set, fv_alpha)
            fv_alpha = fv_alpha.union(fv)
        relation = self.rng.choice(['Equal', 'Less', 'LessEq'])
        if relation == 'Equal':
            relation_formula = Equal(t1, t2)
        elif relation == 'Less':
            relation_formula = Less(t1, t2)
        else:
            relation_formula = LessEq(t1, t2)
        formula = And(alpha_formula, relation_formula)
        return formula, fv_alpha

    def generate_and_with_equality(self, free_vars, size):
        """
        Generate a formula of the form alpha ∧ x = t
        """
        new_size = size - 1
        # Generate alpha
        alpha_formula, fv_alpha = self.generate(free_vars, new_size)
        t = self.random_var(fv = fv_alpha)[0]
        x = self.random_const() #self.random_new_var()
        equality_formula = Equal(x, t)
        formula = And(alpha_formula, equality_formula)
        return formula, fv_alpha.union({t if t.__class__.__name__ == 'Var' else None})
