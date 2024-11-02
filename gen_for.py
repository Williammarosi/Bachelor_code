# !/usr/bin/env python3


import argparse
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
formulaArgs.add_argument("-prob", type=list, default=[0.5], help="probability of operators (not implemented)")
formulaArgs.add_argument("-for_out", type=str, help="output formula file", metavar="<file>")



args = parser.parse_args()

sig, form = main_gen(args.sig, args.pred, args.maxArity, args.size, args.seed)

main_print(sig, form)
if args.o:
    main_file(sig, form)


if args.c:
    if not args.o:
        main_file(sig, form)
    check_monitorability()
