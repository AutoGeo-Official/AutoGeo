import random
import string
from graph import INTERSECT

#####
class TimeoutException(Exception):
    """Custom exception to indicate a timeout."""
    pass

def signal_handler(signum, frame):
    """Signal handler that raises a TimeoutException."""
    raise TimeoutException("Operation timed out due to signal 14 (SIGALRM)")

def remove_uppercase_space(input_string):
    for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ=':
        input_string = input_string.replace(char, "")
    return input_string.replace(" ", "")

#####

def get_ordering_index(first_list, second_list):
    """
    Returns the indices to reorder the second list to match the order of the first list.
    """
    # Create a mapping from element to its index in the first list
    index_map = {element: index for index, element in enumerate(first_list)}

    # Create a list of tuples (index_in_first_list, index_in_second_list)
    ordering = sorted((index_map[element], index) for index, element in enumerate(second_list))

    # Extract and return just the indices from the second list in the correct order
    return [index for _, index in ordering]

def get_wrapped_points(all_points, start, num_points):
    """
    Gets points from the list starting from 'start', wrapping around if necessary,
    to collect 'num_points' items.

    :param all_points: List of all available points.
    :param start: Starting index to pick points.
    :param num_points: Number of points to pick.
    :return: List of points as per specified conditions.
    """
    list_len = len(all_points)  # Length of your points list
    # Use list comprehension and modulo to wrap around
    wrapped_points = [all_points[(start + i) % list_len] for i in range(num_points)]
    return wrapped_points


class ClauseGenerator:
    def __init__(self, defs, clause_relations, is_comma_sep):
        self.defs = defs
        self.defined_points = []
        self.is_comma_sep = is_comma_sep
        self.clause_relations = clause_relations  # list(defs.keys()) or graph.INTERSECT # this is the full set we can't deal with it yet
        # To limit to a few concepts uncomment the following line
        # self.clause_relations = ['triangle', 'parallelogram',]
        self.point_counter = 0  # Start from 0
        self.max_points = 26 * 10  # 26 letters, 10 cycles (0 to 9, inclusive)

    def get_pt_ctr_def_pts(self):
        return self.point_counter, self.defined_points

    def set_pt_ctr_def_pts(self, point_counter, defined_points):
        self.point_counter = point_counter
        self.defined_points = defined_points

    def generate_point(self):
        """
        Generate the next point in sequence: A, B, ..., Z, A1, B1, ..., Z9.
        After Z9, raise an error.
        """
        if self.point_counter >= self.max_points:
            # If we've exhausted all possible names, raise an error
            raise ValueError("All point names have been exhausted.")

        # Calculate the letter and number parts of the name
        letter_part = string.ascii_uppercase[self.point_counter % 26]
        number_part = self.point_counter // 26

        # Prepare the point name
        if number_part == 0:
            # For the first cycle (A-Z), we don't add a number part
            point_name = letter_part
        else:
            # For subsequent cycles, add the number part (reduced by 1 to start from 0)
            point_name = f"{letter_part}{number_part - 1}"

        # Increment the counter for the next call
        self.point_counter += 1

        return point_name

    def generate_new_point(self):
        while True:
            point = self.generate_point()
            if point not in self.defined_points:
                return point

    def choose_random_n_defined_points(self, n):
        """
        Choose n random defined points
        """
        return random.sample(self.defined_points, n)

    def get_text_clause(self, clause_relation, arg_vars, result_vars, all_will_be_defined_pts, result_vars_in_eq):
        """
        Make a canonical clause for a given relation
        """
        pos_new_pts_idx = get_ordering_index(self.defs[clause_relation].construction.args,
                                             self.defs[clause_relation].points + self.defs[clause_relation].args)
        all_inp_pts = result_vars + arg_vars
        all_inp_pts_reordered = [all_inp_pts[pos_new_pts_idx[i]] for i, _ in enumerate(all_inp_pts)]

        if result_vars_in_eq:
            clause_txt = f'{" ".join(all_will_be_defined_pts)} = {clause_relation} {" ".join(all_inp_pts_reordered)}'
        else:
            clause_txt = f'{clause_relation} {" ".join(all_inp_pts_reordered)}'

        # handle special cases
        if clause_relation in ['s_angle', ]:
            clause_txt += f' {random.choice(range(0, 180, 15))}'
        return clause_txt

    def generate_clauses(self, n):
        """
        Generate n random clauses with all points defined
        """
        clauses = []
        if self.is_comma_sep:
            sub_clause_relation = []
            sub_clause_defines_points = []
            sub_clause_needs_defined_points = []
            for i in range(n):
                # choose a random definition key as the first clause
                clause_relation, defines_points, needs_defined_points = self.choose_suitable_clause()
                sub_clause_relation.append(clause_relation)
                # this_clause_must_define = max((0, defines_points - max([0, ] + sub_clause_defines_points)))
                sub_clause_defines_points.append(random.choice(range(defines_points, defines_points + 1)))
                sub_clause_needs_defined_points.append(needs_defined_points)

            defines_points = random.randint(max(sub_clause_defines_points), sum(sub_clause_defines_points))
            all_will_be_defined_pts = self.get_points_that_this_clause_defines(defines_points)

            pts_defined_sp_far = 0
            for i in range(n):
                pts_this_clause_defines = get_wrapped_points(all_will_be_defined_pts, pts_defined_sp_far,
                                                             sub_clause_defines_points[i])
                pts_defined_sp_far += sub_clause_defines_points[i]
                chosen_defined_pts = random.sample(self.defined_points, sub_clause_needs_defined_points[i])
                clause = self.get_text_clause(sub_clause_relation[i], chosen_defined_pts, pts_this_clause_defines,
                                              all_will_be_defined_pts, result_vars_in_eq=(i == 0))
                clauses.append(clause)

            self.add_newly_defined_pts(all_will_be_defined_pts)
            return ', '.join(clauses)
        else:
            for i in range(n):
                # choose a random definition key as the first clause
                clause_relation, defines_points, needs_defined_points = self.choose_suitable_clause()

                chosen_defined_pts = random.sample(self.defined_points, needs_defined_points)
                # Generate names of points that are needed for the clause
                will_be_defined_pts = self.get_points_that_this_clause_defines(defines_points)

                clause = self.get_text_clause(clause_relation, chosen_defined_pts, will_be_defined_pts,
                                              all_will_be_defined_pts=will_be_defined_pts, result_vars_in_eq=True)
                clauses.append(clause)
                self.add_newly_defined_pts(will_be_defined_pts)

            return '; '.join(clauses)

    def get_points_that_this_clause_defines(self, defines_points):
        will_be_defined_pts = []
        while defines_points > 0:
            will_be_defined_pts.append(self.generate_new_point())
            defines_points -= 1
        return will_be_defined_pts

    def add_newly_defined_pts(self, defined_pts):
        self.defined_points += defined_pts

    def choose_suitable_clause(self):
        suitable_clause = False
        while not suitable_clause:
            clause_relation = random.choice(self.clause_relations)
            needs_defined_points = len(self.defs[clause_relation].args)
            defines_points = len(self.defs[clause_relation].points)
            # handle special cases
            if clause_relation in ['s_angle', ]:
                needs_defined_points -= 1
            if needs_defined_points <= len(self.defined_points):
                suitable_clause = True
        return clause_relation, defines_points, needs_defined_points

    def reset(self):
        self.defined_points = []
        self.point_counter = 0


class CompoundClauseGen:
    def __init__(self, definitions, max_sets):
        self.max_sets = max_sets
        self.cg_comma_sep = ClauseGenerator(definitions, INTERSECT, is_comma_sep=True)
        self.cg_single_clause = ClauseGenerator(definitions, list(definitions.keys()), is_comma_sep=False)

    def reset(self):
        self.cg_comma_sep.reset()
        self.cg_single_clause.reset()

    def generate_clauses(self):
        clause_text = ''
        for _ in range(self.max_sets):
            if clause_text != '':
                clause_text += '; '
            clause_text += self.cg_single_clause.generate_clauses(1)

        self.reset()
        return clause_text


if __name__ == "__main__":
    random.seed(3)
    from utils.loading_utils import load_definitions_and_rules

    defs_path = '../defs.txt'
    rules_path = '../rules.txt'

    # Load definitions and rules
    definitions, rules = load_definitions_and_rules(defs_path, rules_path)
    cc_gen = CompoundClauseGen(definitions, 2)

    for j in range(5):
        clause_text = cc_gen.generate_clauses()
        print(clause_text)
