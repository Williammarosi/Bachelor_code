import subprocess

from sig_helper import *
from nfor_helper import *
# from for_helper import *
from log_helper import *

### Todo: Introduce interval_gen/max_leftbound/max_interval, aggr, nolet (gen_fma Arg)
def main_gen(sig_file, num_predicates, max_arity, size, prob, seed, for_file=None):
    """Generate random signature and formula"""
    if not seed:
        seed = random.randint(0, 100000)

    if sig_file:
        signature = f2sig(sig_file)
    else:
        signature = generate_signature(num_predicates, max_arity, seed)
    sig_class = signature[1]

    if for_file:
        with open(for_file, "r") as f:
            formula = f.read()
    else:
        form = FormulaGenerator(signature[1], size, seed, weights=prob)
        formula = form.generate()[0]

    print(f"\n⎯⎯⎯⎯⎯ Seed: {seed} ⎯⎯⎯⎯⎯")
    return sig_class, formula

def main_print(signature, formula):
    """Print signature and formula"""
    sig_str = signature.__2str__()

    if formula.__class__ == str:
        form_str = formula
    else:
        form_str = f"MFOTL Formula:\n{form2str(False, formula)}"
    st = f"\nSignature:\n{sig_str}\n\n{form_str}\n"
    print(st)

def main_file(signature, formula):
    """Write signature and formula to file"""
    sig_str = signature.__2str__()
    form_str = form2str(True, formula)
    with open("test.sig", "w") as f:
        f.write(sig_str)
    print(".sig written to test.sig")
    with open("test.mfotl", "w") as f:
        f.write(form_str)
    print(".mfotl written to test.mfotl")

def main_log(signature, out, i, e, q, r, length, seed):
    """Generate log"""
    if not seed:
        seed = random.randint(0, 100000)
        print(f"⎯⎯⎯⎯⎯Log seed: {seed}⎯⎯⎯⎯⎯")
    generator(signature, out, seed, i, e, q, r, length)
    print(f".CSV written to {out}.csv")
    convert_csv_to_log(out.replace('.log', '.csv'), out.replace('.csv', '.log'))
    print(f".log written to {out}.log")


def check_monitorability():
    """Check if formula is monitorable"""
    sig_file = "test.sig"
    formula_file = "test.mfotl"
    print("⎯"*25)
    # 'monpoly' command:
    monpoly_command = f"monpoly -sig {sig_file} -formula {formula_file} -check"
    try:
        subprocess.run(monpoly_command, shell=True, check=True, stdout=subprocess.PIPE, # capture_output=True,
                            text=True, executable="/bin/zsh")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

def normalize_weights(updated_weights): 
    """
    Normalize the weights of the operators
    Arguments:
        updated_weights: dict
    Returns:
        weights: dict
    """
    weights = {
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
            'Aggreg': 0.1,
            'Let': 0.1
        }
    # Calculate the total weight of updated operators
    fixed_weight = sum(updated_weights.values())
    if fixed_weight > 1:
        print("The sum of the probabilities exceeds 1.")
        for key, value in updated_weights.items():
            updated_weights[key] = value / fixed_weight

    # updated dict
    for key, value in updated_weights.items():
        weights[key] = value

    # total weight of the remaining operators
    remaining_weight = 1 - fixed_weight

    # Get count of unchanged weights
    count_other_weights = len([1 for key, value in weights.items() if key not in updated_weights])

    # Normalize the remaining weights
    for key in weights:
        if key not in updated_weights:
            weights[key] = remaining_weight / count_other_weights

    # remove zero values for readability and negative values in case of wrong input
    weights = {key: value for key, value in weights.items() if value > 0.0}

    return weights
