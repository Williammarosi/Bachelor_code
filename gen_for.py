# !/usr/bin/env python3


import argparse
import json
from call_helper import *


parser = argparse.ArgumentParser(
    prog='rnd_gen',
    description="Generate random signature and formula",
    formatter_class=argparse.MetavarTypeHelpFormatter
    # formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
gensig = parser.add_argument_group("If no signature file is provided")
formulaArgs = parser.add_argument_group("Formula generation arguments")
gensig.add_argument("-pred", type=int, default=4, help="number of predicates if no signature file")
gensig.add_argument("-A", "--maxArity", type=int, default=4, help="maximum arity if no signature file")
gensig.add_argument("-sigout", type=str, help="output signature file", metavar="<file>")

parser.add_argument("-sig", type=str, help="signature file", metavar="<file>")
parser.add_argument("-o", action="store_true", help="write signature and formula to file")
parser.add_argument("-c", action="store_true", help="check monitorability")
parser.add_argument("-seed", type=int, default=None, help="seed for random generation")

formulaArgs.add_argument("-S", "--size", type=int, default=5, help="maximum depth of formula")
formulaArgs.add_argument("-agg", action="store_true", help="allow aggregation operators")
# formulaArgs.add_argument("-prob", type=list, default=[0.5], help="probability of operators (not implemented)")
formulaArgs.add_argument("-for_out", type=str, help="output formula file", metavar="<file>")

prob = parser.add_argument_group("Probability of operators")
prob.add_argument("-prob_dict", type=json.loads, metavar="'dict'",
                  help="dictionary of probabilities of operators. '{\"str1\": float1, \"str2\": float2, ...}' E.g. '{\"and\": 0.5, \"or\": 0.5}'")
prob.add_argument("-prob_and", type=float, help="probability of and operator")
prob.add_argument("-prob_or", type=float, help="probability of or operator")
prob.add_argument("-prob_prev", type=float, help="probability of prev operator")
prob.add_argument("-prob_once", type=float, help="probability of once operator")
prob.add_argument("-prob_since", type=float, help="probability of since operator")
prob.add_argument("-prob_until", type=float, help="probability of until operator")
prob.add_argument("-prob_rand", type=float, help="probability of rand operator")
prob.add_argument("-prob_eand", type=float, help="probability of eand operator")
prob.add_argument("-prob_nand", type=float, help="probability of nand operator")
prob.add_argument("-prob_exists", type=float, help="probability of exists operator")
prob.add_argument("-prob_aggreg", type=float, help="probability of Aggreg operator")


args = parser.parse_args()

weights = {
        'And': args.prob_and,
        'Or': args.prob_or,
        'Prev': args.prob_prev,
        'Once': args.prob_once,
        'Since': args.prob_since,
        'Until': args.prob_until,
        'Rand': args.prob_rand,
        'Eand': args.prob_eand,
        'Nand': args.prob_nand,
        'Exists': args.prob_exists,
    }
### Need to fix the aggregation operator first
if args.agg:
    weights['Aggreg'] = args.prob_aggreg

if args.prob_dict:
    # Check if the provided operators are in the weights
    new_weights = {key.capitalize(): value for key, value in args.prob_dict.items() if key.capitalize() in weights}
    # weights = new_weights # if only the provided operators should be used
    weights = normalize_weights(new_weights)
    print(f"Updated weights: {weights}")
else:
    # Filter out None values
    updated_weights = {key: value for key, value in weights.items() if value is not None}

    if updated_weights == {}:
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
                'Aggreg': 0.1 if args.agg else 0,
                'Let': 0.1
            }
        weights = normalize_weights(weights)
        # print(f"Updated weights: {weights}")
    else:
        weights = normalize_weights(updated_weights)
        # print(f"Updated weights: {weights}")

# print(f"Weights: {weights}")
sig, form = main_gen(args.sig, args.pred, args.maxArity, args.size, weights, args.seed)

main_print(sig, form)
if args.o:
    main_file(sig, form)


if args.c:
    if not args.o:
        main_file(sig, form)
    check_monitorability()
