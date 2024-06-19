import csv
import random


def load_data_from_csv(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter='#')
        for row in reader:
            data.append(row)
    return data


def display_line(row):
    for key, value in row.items():
        if key != 'rnd_states':  # Exclude 'rnd_states' field
            print(f"{key}: {value}")
    print()


def plot(problem_txt):
    import sys
    sys.path.append('..')
    import graph as gh
    import problem as pr
    from utils.loading_utils import load_definitions_and_rules
    defs_path = '../defs.txt'
    rules_path = '../rules.txt'

    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)
    p = pr.Problem.from_txt(problem_txt)
    g, _ = gh.Graph.build_problem(p, definitions)
    gh.nm.draw(
        g.type2nodes[gh.Point],
        g.type2nodes[gh.Line],
        g.type2nodes[gh.Circle],
        g.type2nodes[gh.Segment])


def main():
    filename = '../data/nl_fl_dataset.csv'
    data = load_data_from_csv(filename)
    total_lines = len(data)
    current_line = random.randint(0, total_lines - 1)

    # Keyboard commands:
    # 'n' - next line
    # 'p' - plot the current line
    # 'f' - previous (former) line

    should_print = True
    while True:
        if should_print:
            print(f"Current line: {current_line + 1} / {total_lines}")
            display_line(data[current_line])

        user_input = input("Enter 'n' for next line, 'p' for previous line, or line number to display that line: ")
        should_print = True
        if user_input == 'n':
            current_line = (current_line + 1) % total_lines
        elif user_input == 'f':
            current_line = (current_line - 1) % total_lines
        elif user_input == 'p':
            plot(data[current_line]['fl_statement'])
            should_print = False
        elif user_input.isdigit():
            line_number = int(user_input) - 1
            if 0 <= line_number < total_lines:
                current_line = line_number
            else:
                print("Invalid line number. Please enter a valid line number.")
        else:
            print("Invalid input. Please enter 'n' for next line, 'p' for previous line, or line number.")


if __name__ == "__main__":
    main()
