import random
import pandas as pd
# import numpy as np

def generator(sig, out, seed, i=10, e=None, q=20, r=0.2, length=300):
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
    data_value_cache = []
    for _ in range(q):
        data_value_cache.append(random.randrange(100000000, 1000000000))

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
                        data_value = random.randrange(100000000, 1000000000)
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




def convert_csv_to_log(input_csv_path, output_log_path):
    with open(input_csv_path, mode='r') as csv_file, open(output_log_path, mode='w') as log_file:
        for row in csv_file:
            # print(row.split(','))
            row = row.split(',')
            eventtype = row[0].strip()
            timestamp = row[2].strip().replace('ts=', '')

            args = [row[i+3].strip().replace(f'x{i}=', '') for i in range(len(row)-3)]
            args = [arg for arg in args if arg]

            log_entry = f"@{timestamp} {eventtype}({','.join(args)});"
            log_file.write(log_entry + '\n')

    # df = pd.read_csv(input_csv_path)
