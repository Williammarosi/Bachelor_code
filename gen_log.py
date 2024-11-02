#!/usr/bin/env python3

import sys
import argparse

from call_helper import *


log_parser = argparse.ArgumentParser(
    prog='rnd_log',
    description="Generate logs",
    # formatter_class=argparse.MetavarTypeHelpFormatter
    # formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
sig_parser = log_parser.add_mutually_exclusive_group()
sig_parser.add_argument("-sig", type=str, help="signature file", metavar="<file>")
# sig_arg = sig_parser.add_argument_group("If no signature file is provided")
sig_parser.add_argument("-rnd_sig", nargs=2, default=[4, 4], help="pred arity", metavar="<int>")

form_parser = log_parser.add_mutually_exclusive_group()
form_parser.add_argument("-form", type=str, help="formula file", metavar="<file>")
form_parser.add_argument("-rnd_form", type=int, default=5, help="size of formula", metavar="<int>")

log_parser.add_argument("-i", type=int, default=10, help="indexrate", metavar="<int>")
log_parser.add_argument("-e", type=int, default=log_parser.get_default(''), help="eventrate", metavar="<int>")
log_parser.add_argument("-r", type=float, default=0.2, help="ratio of new values 0.0-1.0", metavar="<float>")
log_parser.add_argument("-q", type=int, default=10, help="number of queries", metavar="<int>")
log_parser.add_argument("-l", "--len", type=int, default=20, help="length of log", metavar="<int>")
log_parser.add_argument("-log", type=str, default="test.csv", help="output log file", metavar="<int>")
log_parser.add_argument("-logseed", type=int, default=None, help="seed for log generation", metavar="<int>")

extra = log_parser.add_argument_group("Extra options")
extra.add_argument("-seed", type=int, default=None, help="seed for random generation", metavar="<int>")

args = log_parser.parse_args()

if (not args.sig) and args.form:
    print("Error: Signature file is required if formula file is provided. \n -h   Show help message\nExiting...")
    sys.exit(1)

sig, form = main_gen(args.sig, args.rnd_sig[0], args.rnd_sig[1], args.rnd_form, args.seed, forFile=args.form)

main_print(sig, form)

if args.log:
    main_log(sig, out=args.log, i=args.i, e=args.e, r=args.r, q=args.q, length=args.len, seed=args.logseed)
    # gen.to_csv('log.csv', index=False, header=False) # , sep=' '
