import csv
import argparse
from collections import Counter
from math import log2
from graphviz import Digraph


class DecisionTree:
    def __init__(self, root_node, header):
        self.root_node = root_node
        self.header = header

    def __str__(self, indent=""):
        return self.root_node.__str__(headings=self.header, indent=indent)

    @staticmethod
    def _eval_fn(criterion):
        if criterion == "entropy":
            return entropy
        elif criterion == "gini":
            return gini
        else:
            raise ValueError(f"Unknown criterion: {criterion}")

    @classmethod
    def train(cls, data, header, criterion="entropy"):
        """Trains a decision tree.

        Args:
            data (list of lists): The training dataset.
            header (list): The list of feature names.
            criterion (str): The criterion to use for splitting ('entropy' or 'gini').
        """
        eval_fn = DecisionTree._eval_fn(criterion)
        root = cls._grow_tree(data, criterion=eval_fn)
        return cls(root, header)

    @staticmethod
    def _grow_tree(rows, criterion):
        """Grows and then returns a binary decision tree node."""
        if not rows:
            return DecisionNode()
        currentScore = criterion(rows)

        bestGain = 0.0
        bestAttribute = None
        bestSets = None

        columnCount = len(rows[0]) - 1
        for col in range(columnCount):
            columnValues = set(row[col] for row in rows)
            for value in columnValues:
                (set1, set2) = divideSet(rows, col, value)
                if not set1 or not set2:
                    continue

                p = len(set1) / len(rows)
                gain = currentScore - p * criterion(set1) - (1 - p) * criterion(set2)
                if gain > bestGain:
                    bestGain = gain
                    bestAttribute = (col, value)
                    bestSets = (set1, set2)

        summary = {"impurity": currentScore, "samples": len(rows)}

        if bestGain > 0:
            trueBranch = DecisionTree._grow_tree(bestSets[0], criterion)
            falseBranch = DecisionTree._grow_tree(bestSets[1], criterion)
            return DecisionNode(
                col=bestAttribute[0],
                value=bestAttribute[1],
                trueBranch=trueBranch,
                falseBranch=falseBranch,
                summary=summary,
            )
        else:
            return DecisionNode(class_counts=uniqueCounts(rows), summary=summary)

    def classify(self, sample, handle_missing=True):
        if handle_missing:
            return self.root_node.classify_with_missing_data(sample)
        else:
            return self.root_node.classify(sample)

    def prune(self, min_gain, criterion="entropy", notify=False):
        eval_fn = DecisionTree._eval_fn(criterion)
        self.root_node.prune(min_gain, criterion=eval_fn, notify=notify)

    def export_graph(self, filename):
        """Exports the decision tree to a file using Graphviz with deterministic output."""
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
                tree.trueBranch, header, dot, counter
            )
            dot.edge(node_id, true_child_id, label="True")

            false_child_id = DecisionTree._export_tree(
                tree.falseBranch, header, dot, counter
            )
            dot.edge(node_id, false_child_id, label="False")

        # 4. Return the ID of the current node so its parent can create an edge to it.
        return node_id


def divideSet(rows, column, value):
    """Splits a dataset on a specific column.

    Args:
        rows (list of lists): The dataset.
        column (int): The index of the column to test.
        value (int, float, or str): The value to split on.

    Returns:
        tuple: A tuple containing two lists of rows.
    """
    splittingFunction = None
    if isinstance(value, (int, float)):  # for numeric values
        splittingFunction = lambda row: row[column] >= value
    else:  # for strings
        splittingFunction = lambda row: row[column] == value
    list1 = [row for row in rows if splittingFunction(row)]
    list2 = [row for row in rows if not splittingFunction(row)]
    return (list1, list2)


def uniqueCounts(rows):
    """Counts the occurrences of each class in the dataset."""
    # The class label is in the last column
    return Counter(row[-1] for row in rows)


def entropy(rows):
    """Calculates the entropy for a list of rows
    https://en.wikipedia.org/wiki/Entropy_(information_theory)
    """
    if not rows:
        return 0.0
    total = len(rows)
    return -sum(
        (count / total) * log2(count / total) for count in uniqueCounts(rows).values()
    )


def gini(rows):
    """Calculates the Gini impurity for a list of rows.
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    if not rows:
        return 0.0
    total = len(rows)
    return 1.0 - sum((count / total) ** 2 for count in uniqueCounts(rows).values())


class DecisionNode:
    def __init__(
        self,
        col=None,
        value=None,
        class_counts=None,
        trueBranch=None,
        falseBranch=None,
        summary=None,
    ):
        self.col = col
        self.value = value
        self.class_counts = class_counts  # only for leaves; None for internal nodes
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch
        self.summary = summary

    def __str__(self, headings=None, indent=""):
        if self.class_counts:  # Leaf
            lsX = sorted(self.class_counts.items())
            return ", ".join(f"{x}: {y}" for x, y in lsX)
        else:
            if headings:
                szCol = headings[self.col]
            else:
                szCol = f"Column {self.col}"

            # szCol = f"Column {self.col}"
            # if headings and szCol in headings:
            #     szCol = headings[szCol]

            if isinstance(self.value, (int, float)):
                decision = f"{szCol} >= {self.value}?"
            else:
                decision = f"{szCol} == {self.value}?"

            trueBranchStr = (
                indent + "yes -> " + self.trueBranch.__str__(headings, indent + "    ")
            )
            falseBranchStr = (
                indent + "no  -> " + self.falseBranch.__str__(headings, indent + "    ")
            )
            return decision + "\n" + trueBranchStr + "\n" + falseBranchStr

    def pick_branch(self, v):
        cond = v >= self.value if isinstance(v, (int, float)) else v == self.value
        return self.trueBranch if cond else self.falseBranch

    def classify(self, observations):
        """Classifies the observations - assume no missing observations"""
        if self.class_counts:  # leaf
            return self.class_counts
        v = observations[self.col]
        branch = self.pick_branch(v)
        return branch.classify(observations)

    def classify_with_missing_data(self, observations):
        """Classifies the observations - may have missing (None) observations"""
        if self.class_counts:  # leaf
            return self.class_counts

        v = observations[self.col]
        if v is not None:
            branch = self.pick_branch(v)
            return branch.classify_with_missing_data(observations)

        # Handle missing feature: combine results from both branches,
        # weighted by the number of samples in each branch's subtree.
        tr = self.trueBranch.classify_with_missing_data(observations)
        fr = self.falseBranch.classify_with_missing_data(observations)
        tcount = sum(tr.values())
        fcount = sum(fr.values())
        tw = tcount / (tcount + fcount)
        fw = fcount / (tcount + fcount)
        keys = tr.keys() | fr.keys()
        return Counter({k: tr.get(k, 0) * tw + fr.get(k, 0) * fw for k in keys})

    def prune(self, minGain, criterion=entropy, notify=False):
        """Prunes the obtained tree according to the minimal gain (entropy or Gini)."""
        # recursive call for each branch
        if not self.trueBranch.class_counts:
            self.trueBranch.prune(minGain, criterion, notify)
        if not self.falseBranch.class_counts:
            self.falseBranch.prune(minGain, criterion, notify)

        # merge leaves (potentionally)
        if self.trueBranch.class_counts and self.falseBranch.class_counts:
            tb, fb = [], []

            for v, c in self.trueBranch.class_counts.items():
                tb += [[v]] * c
            for v, c in self.falseBranch.class_counts.items():
                fb += [[v]] * c

            p = len(tb) / len(tb + fb)
            delta = criterion(tb + fb) - p * criterion(tb) - (1 - p) * criterion(fb)
            if delta < minGain:
                if notify:
                    print("A branch was pruned: gain = %f" % delta)
                self.trueBranch, self.falseBranch = None, None
                self.class_counts = uniqueCounts(tb + fb)


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


def loadCSV(file):
    """Loads a CSV file, reads the header, and converts data types."""

    def convertTypes(s):
        s = s.strip()
        try:
            return float(s) if "." in s else int(s)
        except ValueError:
            return s

    with open(file, "rt") as f:
        reader = csv.reader(f)
        header = next(reader)  # Read the first line as the header
        data = [[convertTypes(item) for item in row] for row in reader]
        return header, data


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
    #     4. Plot the decision tree
    #     5. Classify without missing data
    #     6. Classify with missing data
    #     (7.) Prune the decision tree according to a minimal gain level
    #     (8.) Plot the pruned tree

    decisionTree = None
    header = None
    trainingData = None

    if args.example == 1:
        # the smaller example
        header, trainingData = loadCSV("data/tbc.csv")
        dt = DecisionTree.train(trainingData, header, criterion=args.criterion)
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
    else:
        # the bigger example
        header, trainingData = loadCSV("data/fishiris.csv")  # demo data from matlab
        dt = DecisionTree.train(trainingData, header, criterion=args.criterion)
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

    if args.plot:
        dt.export_graph(args.plot)
