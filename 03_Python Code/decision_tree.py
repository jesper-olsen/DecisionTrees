import csv
import argparse
from collections import Counter
from math import log2
from graphviz import Digraph
from typing import List, Dict, Optional, Union, Tuple

type Sample = List[Union[int, float, str, None]]

class DecisionTree:
    def __init__(self, root_node: 'DecisionNode', header: List[str]):
        self.root_node = root_node
        self.header = header

    def __str__(self, indent=""):
        return self.root_node.__str__(headings=self.header, indent=indent)

    @staticmethod
    def _eval_fn(criterion):
        match criterion:
            case "entropy": return entropy
            case "gini": return gini
            case _: raise ValueError(f"Unknown criterion: {criterion}")

    @classmethod
    def train(
        cls,
        data: List[Sample],
        header: List[str],
        criterion: str = "entropy",
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
    ) -> "DecisionTree":
        """Trains a decision tree.

        Args:
            data (list of lists): The training dataset.
            header (list): The list of feature names.
            criterion (str): The criterion to use for splitting ('entropy' or 'gini').
        """
        eval_fn = DecisionTree._eval_fn(criterion)
        root = cls._grow_tree(
            data, min_samples_split, max_depth, criterion=eval_fn, depth=0
        )
        return cls(root, header)

    @staticmethod
    def _grow_tree(rows, min_samples_split, max_depth, criterion, depth):
        """Grows and then returns a binary decision tree node."""
        if not rows:
            return DecisionNode()

        currentScore = criterion(count_classes(rows))

        # Pre-pruning
        summary = {"impurity": currentScore, "samples": len(rows)}
        if (max_depth is not None and depth >= max_depth) or (
            len(rows) < min_samples_split
        ):
            return DecisionNode(class_counts=count_classes(rows), summary=summary)

        best_gain = 0.0
        best_rule = None
        best_sets = None

        column_count = len(rows[0]) - 1
        for col in range(column_count):
            column_values = set(row[col] for row in rows)
            for value in column_values:
                (set1, set2) = split_set(rows, col, value)
                if not set1 or not set2:
                    continue

                p = len(set1) / len(rows)
                gain = (
                    currentScore
                    - p * criterion(count_classes(set1))
                    - (1 - p) * criterion(count_classes(set2))
                )
                if gain > best_gain:
                    best_gain = gain
                    best_rule = (col, value)
                    best_sets = (set1, set2)

        summary = {"impurity": currentScore, "samples": len(rows)}

        if best_gain > 0:
            true_branch = DecisionTree._grow_tree(
                best_sets[0], min_samples_split, max_depth, criterion, depth + 1
            )
            false_branch = DecisionTree._grow_tree(
                best_sets[1], min_samples_split, max_depth, criterion, depth + 1
            )
            return DecisionNode(
                col=best_rule[0],
                value=best_rule[1],
                true_branch=true_branch,
                false_branch=false_branch,
                summary=summary,
            )
        else:
            return DecisionNode(class_counts=count_classes(rows), summary=summary)

    def classify(self, sample: Sample, handle_missing=True):
        if handle_missing:
            return self.root_node.classify_with_missing_data(sample)
        else:
            return self.root_node.classify(sample)

    def prune(self, min_gain, criterion="entropy", notify=False):
        eval_fn = DecisionTree._eval_fn(criterion)
        self.root_node.prune(min_gain, criterion=eval_fn, notify=notify)

    def export_graph(self, filename):
        """Exports the decision tree to a file using Graphviz with deterministic output."""
        if not "." in filename:
            filename += ".png"
        file, ext = filename.rsplit(".", 1)

        dot = Digraph()
        dot.attr("node", shape="box", style="filled, rounded")

        # We use a mutable list as a counter so it can be modified by the recursive calls.
        # Each node gets a unique, sequential ID (0, 1, 2, ...).
        node_counter = [0]

        self._export_tree(self.root_node, self.header, dot, node_counter)
        dot.render(file, format=ext, cleanup=True)
        print(f"Decision tree exported to {filename}")

    @staticmethod
    def _export_tree(tree, header, dot, counter):
        """
        Recursively builds the Graphviz DOT representation with deterministic node IDs.
        Returns the ID of the node it just processed.
        """
        # 1. Assign the current node a unique ID from the counter.
        node_id = str(counter[0])
        counter[0] += 1

        # 2. Create the node's label based on whether it's a leaf or decision node.
        if tree.class_counts:  # Leaf node
            class_counts = "<BR/>".join(
                f"{k}: {v}" for k, v in sorted(tree.class_counts.items())
            )
            label = f"""<
    <B>Leaf</B><BR/>
    {class_counts}<BR/>
    impurity = {tree.summary['impurity']:.3f}<BR/>
    samples = {tree.summary['samples']}
    >"""
            dot.node(node_id, label, fillcolor="#e58139aa")
        else:  # Decision node
            column_name = header[tree.col]
            condition = (
                f"{column_name} &ge; {tree.value}"
                if isinstance(tree.value, (int, float))
                else f"{column_name} == {tree.value}"
            )
            label = f"""<
    <B>{condition}</B><BR/>
    impurity = {tree.summary['impurity']:.3f}<BR/>
    samples = {tree.summary['samples']}
    >"""
            dot.node(node_id, label, fillcolor="#399de5aa")

            # 3. Recurse for children and create edges to them.
            # The child's ID is returned by the recursive call.
            true_child_id = DecisionTree._export_tree(
                tree.true_branch, header, dot, counter
            )
            dot.edge(node_id, true_child_id, label="True")

            false_child_id = DecisionTree._export_tree(
                tree.false_branch, header, dot, counter
            )
            dot.edge(node_id, false_child_id, label="False")

        # 4. Return the ID of the current node so its parent can create an edge to it.
        return node_id


def split_set(rows, column, value):
    """Splits a dataset on a specific column.

    Args:
        rows (list of lists): The dataset.
        column (int): The index of the column to test.
        value (int, float, or str): The value to split on.

    Returns:
        tuple: A tuple containing two lists of rows.
    """
    set1, set2 = [], []
    for row in rows:
        v = row[column]
        if v is None:
            # missing value - duplicate row into both sets
            set1.append(row)
            set2.append(row)
        elif isinstance(value, (int, float)):
            (set1 if v >= value else set2).append(row)
        else:
            (set1 if v == value else set2).append(row)
    return set1, set2


def count_classes(rows) -> Counter:
    """Counts the occurrences of each class in the dataset."""
    # The class label is in the last column
    return Counter(row[-1] for row in rows)


def entropy(counts: Counter) -> float:
    """Calculates entropy from a Counter object of class counts.
    https://en.wikipedia.org/wiki/Entropy_(information_theory)"""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * log2(c / total) for c in counts.values())


def gini(counts: Counter) -> float:
    """Calculates Gini impurity from a Counter object of class counts.
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity"""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return 1.0 - sum((c / total) ** 2 for c in counts.values())


class DecisionNode:
    def __init__(
        self,
        col=None,
        value=None,
        class_counts=None,
        true_branch=None,
        false_branch=None,
        summary=None,
    ):
        self.col = col
        self.value = value
        self.class_counts = class_counts  # only for leaves; None for internal nodes
        self.true_branch = true_branch
        self.false_branch = false_branch
        self.summary = summary

    def __str__(self, headings=None, indent=""):
        if self.class_counts:  # Leaf
            sorted_counts = sorted(self.class_counts.items())
            return ", ".join(f"{x}: {y}" for x, y in sorted_counts)
        else:
            column_name = headings[self.col] if headings else f"Column {self.col}"

            if isinstance(self.value, (int, float)):
                decision = f"{column_name} >= {self.value}?"
            else:
                decision = f"{column_name} == {self.value}?"

            true_branchStr = (
                indent + "yes -> " + self.true_branch.__str__(headings, indent + "    ")
            )
            false_branchStr = (
                indent
                + "no  -> "
                + self.false_branch.__str__(headings, indent + "    ")
            )
            return decision + "\n" + true_branchStr + "\n" + false_branchStr

    def pick_branch(self, v):
        cond = v >= self.value if isinstance(v, (int, float)) else v == self.value
        return self.true_branch if cond else self.false_branch

    def classify(self, sample: Sample):
        """Classifies the sample - assume no missing features"""
        if self.class_counts:  # leaf
            return self.class_counts
        v = sample[self.col]
        branch = self.pick_branch(v)
        return branch.classify(sample)

    def classify_with_missing_data(self, sample: Sample):
        """Classifies the sample - may have missing (None) features"""
        if self.class_counts:  # leaf
            return self.class_counts

        v = sample[self.col]
        if v is not None:
            branch = self.pick_branch(v)
            return branch.classify_with_missing_data(sample)

        # Handle missing feature: combine results from both branches,
        # weighted by the number of samples in each branch's subtree.
        tr = self.true_branch.classify_with_missing_data(sample)
        fr = self.false_branch.classify_with_missing_data(sample)
        tcount = sum(tr.values())
        fcount = sum(fr.values())
        total_count = tcount + fcount

        if total_count == 0:
            return Counter()

        tw = tcount / total_count
        fw = fcount / total_count
        keys = tr.keys() | fr.keys()
        return Counter({k: tr.get(k, 0) * tw + fr.get(k, 0) * fw for k in keys})

    def prune(self, minGain, criterion=entropy, notify=False):
        """Prunes the obtained tree according to the minimal gain (entropy or Gini)."""
        # recursive call for each branch
        if not self.true_branch.class_counts:
            self.true_branch.prune(minGain, criterion, notify)
        if not self.false_branch.class_counts:
            self.false_branch.prune(minGain, criterion, notify)

        # merge leaves (potentionally)
        if self.true_branch.class_counts and self.false_branch.class_counts:
            true_counts = self.true_branch.class_counts
            false_counts = self.false_branch.class_counts
            merged_counts = true_counts + false_counts
            merged_impurity = criterion(merged_counts)
            total_samples = sum(merged_counts.values())
            p_true = sum(true_counts.values()) / total_samples
            child_impurity = p_true * criterion(true_counts) + (1 - p_true) * criterion(
                false_counts
            )
            gain = merged_impurity - child_impurity
            if gain < minGain:
                if notify:
                    print(f"A branch was pruned: gain = {gain:.4f}")
                self.true_branch, self.false_branch = None, None
                self.class_counts = merged_counts


def print_classification_result(sample, result):
    """Prints a formatted and human-readable classification result."""
    if not result:
        print(f"Could not classify sample: {sample}")
        return

    # Determine the final prediction by finding the class with the highest score
    prediction = max(result, key=result.get)

    # Sort the results by score (descending) for deterministic and clear output
    sorted_results = sorted(result.items(), key=lambda item: item[1], reverse=True)

    print("-" * 40)
    print(f"Input Sample: {sample}")
    print(f"--> Predicted Class: '{prediction}'")
    print("\nDetailed Scores (Leaf Node Counts / Weights):")
    for class_name, score in sorted_results:
        if isinstance(score, float):
            # For missing data, these are weighted counts
            print(f"    - {class_name:<12}: {score:.4f}")
        else:
            # For complete data, these are direct sample counts
            print(f"    - {class_name:<12}: {score}")
    print("-" * 40)


def load_csv(fname: str) -> Tuple[List[str],List[Sample]]:
    """Loads a CSV file, reads the header, and converts data types."""

    def convertTypes(s):
        s = s.strip()
        if s == "?":  # treat '?' as missing
            return None
        try:
            return float(s) if "." in s else int(s)
        except ValueError:
            return s

    with open(fname, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)  # Read the first line as the header
        data = [[convertTypes(item) for item in row] for row in reader]
        return header, data

def small_example(args):
    header, trainingdata = loadCSV("data/tbc.csv")
    dt = DecisionTree.train(trainingdata, header, criterion=args.criterion)
    print(dt)

    print("\n--- Classification Examples ---")

    # Example 1: A sample with complete data
    complete_sample = ["ohne", "leicht", "Streifen", "normal", "normal"]
    result1 = dt.classify(complete_sample, handle_missing=False)
    print_classification_result(complete_sample, result1)

    missing_sample = [None, "leicht", None, "Flocken", "fiepend"]
    result2 = dt.classify(missing_sample, handle_missing=True)
    print_classification_result(missing_sample, result2)  # no longer unique
    # Don't forget if you compare the resulting tree with the tree in my presentation: here it is a binary tree!
    if args.plot: dt.export_graph(args.plot)

def bigger_example(args):
    header, trainingdata = load_csv("data/fishiris.csv")  # demo data from matlab
    dt = DecisionTree.train(trainingdata, header, criterion=args.criterion)
    print(dt)
    # notify, when a branch is pruned (one time in this example)
    dt.prune(0.5, criterion=args.criterion, notify=True)
    print(dt)

    print("\n--- Classification Examples ---")

    # Example 1: A sample with complete data
    complete_sample = [6.0, 2.2, 5.0, 1.5]
    result1 = dt.classify(complete_sample, handle_missing=False)
    print_classification_result(complete_sample, result1)

    # Example 2: A sample with missing data
    missing_sample = [None, None, None, 1.5]
    result2 = dt.classify(missing_sample, handle_missing=True)
    print_classification_result(missing_sample, result2)  # no longer unique
    if args.plot: dt.export_graph(args.plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decision tree demo")
    parser.add_argument(
        "example",
        type=int,
        choices=[1, 2],
        help="Which example to run: 1=small dataset (tbc), 2=larger dataset (fishiris)",
    )
    parser.add_argument(
        "--criterion",
        choices=["entropy", "gini"],
        default="entropy",
        help="Criterion to use for splits (default: entropy)",
    )
    parser.add_argument(
        "--plot",
        metavar="FILE",
        help="Export the decision tree as a Graphviz image (e.g. tree.png or tree.svg)",
    )
    args = parser.parse_args()

    # All examples do the following steps:
    #     1. Load training data
    #     2. Let the decision tree grow
    #     4. Print the decision tree to stdout
    #     5. Classify without missing data
    #     6. Classify with missing data
    #     (7.) Prune the decision tree according to a minimal gain level
    #     (8.) Plot the pruned tree

    match args.example:
        case 1: small_example(args)
        case 2: bigger_example(args)
        case _: print("Invalid example - should be 1 or 2")
