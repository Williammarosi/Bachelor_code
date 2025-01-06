import random
import pandas as pd
import itertools
# import numpy as np

def generator(sig, out, seed, i=10, e=None, q=20, r=0.2, length=300, int_range = None):
    """
    Generate a random log for the given signature.
    
    - sig: signature
    - i: Index rate (time-points pr time-stamp).
    - e: Event rate (events pr time-stamp).
    - q: Number of most recently sampled unique data values.
    - r: Probability to sample a fresh data value.
    - length: Total time-wise length of the log.

    Returns:
    - pandas DataFrame
    """
    # Set seed
    # if seed is None:
    #     seed = random.randint(0, 1000000)
    #     print(f"Log Seed: {seed}")
    random.seed(seed)

    # Initialize the recently used data values
    print(int_range)
    if int_range is None:
        int_range = (100000000,1000000000)
    data_value_cache = []
    for _ in range(q):
        data_value_cache.append(random.randrange(int_range[0], int_range[1]))

    if e is None or e < i:
        e = i
    events = []
    tp = 0
    for time_stamp in range(0, length):
        for _ in range(i):
            for _ in range(e//i):
                # Randomly select predicate
                event_type = random.choice(sig.predicates)
                # print(event_type.name)
                # print([', '.join('int' for v in range(event_type.len))])#, event_type.variables)
                # arg_types = sig[event_type]
                
                arg_values = []

                for _ in range(event_type.len):
                    if random.random() < r:
                        # Generate a fresh data value
                        data_value = random.randrange(int_range[0], int_range[1])
                        data_value_cache.append(data_value)

                        if len(data_value_cache) > q:
                            data_value_cache.pop(0)
                    else:
                        data_value = random.choice(data_value_cache)
                    arg_values.append(f"x{len(arg_values)}={data_value}")

                event = {
                    'eventtype': event_type.name,
                    'time-point': f"tp={tp}", #int(time_stamp * i + a)}",
                    'time-stamp': f"ts={time_stamp}"
                }

                for idx, arg_value in enumerate(arg_values):
                    event[f'arg{idx}'] = arg_value

                events.append(event)
            tp += 1

    # Create DataFrame
    df = pd.DataFrame(data = events)
    # Write to file without trailing commas
    with open(out.replace('.log', '.csv'), "w") as f:
        f.write("\n".join([l.strip(", ") for l in df.to_csv(index=False, header=None).split("\n")]))
    convert_csv_to_log(out, out.replace('.csv', '.log'))




# def convert_csv_to_log(input_csv_path, output_log_path):
#     with open(input_csv_path, mode='r') as csv_file, open(output_log_path, mode='w') as log_file:
#         print(f"testing: {csv_file}")
#         current_timepoint = 0
#         for row in csv_file:
#             # print(row.split(','))
#             row = row.split(',')
#             eventtype = row[0].strip()
#             timepoint = row[1].strip().replace('tp=', '')
#             timestamp = row[2].strip().replace('ts=', '')

#             args = [row[i+3].strip().replace(f'x{i}=', '') for i in range(len(row)-3)]
#             args = [arg for arg in args if arg]

            
#             if timepoint != current_timepoint:
#                 log_entry = f"@{timestamp} {eventtype}({','.join(args)})"
#                 current_timepoint = timepoint
#             log_file.write(log_entry + '\n')

    # df = pd.read_csv(input_csv_path)

def convert_csv_to_log(input_csv_path, output_log_path):
    # Dictionary to group events by timestamp
    # Key: timestamp (string or float), Value: list of "(EventType(args))" strings
    events_by_timestamp = {}

    with open(input_csv_path, mode='r') as csv_file:
        for row in csv_file:
            row = row.strip().split(',')
            eventtype = row[0].strip()         # e.g., "P1"
            timepoint = row[1].strip().replace('tp=', '')
            timestamp = row[2].strip().replace('ts=', '')  # e.g., "0"

            # Collect argument values
            args = []
            # Start from index 3, because the row is: [eventtype, timepoint, timestamp, x0=..., x1=..., ...]
            for i in range(3, len(row)):
                arg_raw = row[i].strip()
                # Each arg should look like x0=29 or x1=82, or might be empty
                if arg_raw:
                    # Remove "xN=" prefix if it exists
                    splitted = arg_raw.split('=')
                    if len(splitted) == 2:
                        _, val = splitted
                        args.append(val.strip())

            # Build the event string, e.g. "P1(29,82)" or "P2(86)" or "P0()"
            if args:
                event_str = (eventtype, ','.join(args))
                # event_str = f"{eventtype}({','.join(args)})"
            else:
                event_str = (eventtype, '')
                # event_str = f"{eventtype}()"

            # Group events by timestamp in the dictionary
            if timestamp not in events_by_timestamp:
                events_by_timestamp[timestamp] = {}
                events_by_timestamp[timestamp][timepoint] = [event_str]
            else:
                if timepoint not in events_by_timestamp[timestamp]:
                    events_by_timestamp[timestamp][timepoint] = [event_str]
                else:
                    events_by_timestamp[timestamp][timepoint].append(event_str)
    # Now write them to output
    with open(output_log_path, mode='w') as log_file:
        # Sort timestamps if you want chronological order
        # Or just iterate in insertion order if Python 3.7+ (dict is ordered by default).
        for ts in sorted(events_by_timestamp.keys(), key=lambda x: float(x)):
            # Combine all events for this timestamp in one line
            
            # e.g. @0 P1(29,82) P2(86);
            for timepoint, events in events_by_timestamp[ts].items():
                line_str = f"@{ts} "
                # print(events)
                a = itertools.groupby(sorted(events, key=lambda x: x[0]), lambda x: x[0])
                for key, subiter in a:
                    
                    line_str += f"{key}{''.join(f'({item[1]})' for item in subiter)} "
                    # print(line_str)
                # print(a)
                events_list = sorted(set(event[0] for event in events))
                event_dir = {}
                
                # events = events_list#[event[0] for event in events if event]
                
                # line_str = f"@{ts} " + " ".join([x + [f"({ev[0]})" for ev in events if ev[0] == x else ""] for x in events_list]) + ";"
                log_file.write(line_str[:-1] + ";\n")
            # events_list = events_by_timestamp[ts]
            # line_str = f"@{ts} " + " ".join(events_list) + ";"
            # log_file.write(line_str + "\n")