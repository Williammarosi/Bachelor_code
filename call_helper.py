import subprocess

from sig_helper import *
from for_helper import *
from log_helper import *

### Todo: Introduce interval_gen/max_leftbound/max_interval, aggr, nolet (gen_fma Arg)
def main_gen(sigFile, num_predicates, max_arity, size, seed, forFile=None):
    """Generate random signature and formula"""
    if not seed:
        seed = random.randint(0, 100000)
        print(f"\n–––––Seed: {seed}–––––")

    if sigFile:
        signature = f2sig(sigFile)
    else:
        signature = generate_signature(num_predicates, max_arity, seed)
        sigClass = signature[1]

    if forFile:
        with open(forFile, "r") as f:
            formula = f.read()
    else:
        form = FormulaGenerator(signature[1], size, seed)
        formula = form.generate()[0]
    return signature[1], formula

def main_print(signature, formula):
    """Print signature and formula"""
    sig_str = signature.__str__()

    if formula.__class__ == str:
        form_str = formula
    else:
        form_str = f"MFOTL Formula:\n{form2str(False, formula)}"
    st = f"\nSignature:\n{sig_str}\n\n{form_str}\n"
    print(st)

def main_file(signature, formula):
    """Write signature and formula to file"""
    sig_str = signature.__str__()
    form_str = form2str(True, formula)
    with open("test.sig", "w") as f:
        f.write(sig_str)
    print(f".sig written to test.sig")
    with open("test.mfotl", "w") as f:
        f.write(form_str)
    print(f".mfotl written to test.mfotl")

def main_log(signature, out, i, e, q, r, length, seed):
    """Generate log"""
    if not seed:
        seed = random.randint(0, 100000)
        print(f"–––––Log seed: {seed}–––––")
    generator(signature, out, seed, i, e, q, r, length)
    print(f".CSV written to {out}.csv")
    convert_csv_to_log(out.replace('.log', '.csv'), out.replace('.csv', '.log'))
    print(f".log written to {out}.log")


def check_monitorability():
    """Check if formula is monitorable"""
    sig_file = "test.sig"
    formula_file = "test.mfotl"
    print()
    # 'monpoly' command:
    monpoly_command = f"monpoly -sig {sig_file} -formula {formula_file} -check"
    result = subprocess.run(monpoly_command, shell=True, check=True, stdout=subprocess.PIPE, # capture_output=True,
                            text=True, executable="/bin/zsh")
