import csv
import argparse
from collections import Counter
from math import log2
from graphviz import Digraph


class DecisionNode:
    def __init__(
        self, col=None, value=None, results=None, trueBranch=None, falseBranch=None
    ):
        self.col = col
        self.value = value
        self.results = results  # None for nodes, not None for leaves
        self.trueBranch = trueBranch
        self.falseBranch = falseBranch

    def __str__(self, headings=None, indent=""):
        if self.results is not None:  # Leaf
            lsX = sorted(self.results.items())
            return ", ".join(f"{x}: {y}" for x, y in lsX)
        else:
            szCol = f"Column {self.col}"
            if headings and szCol in headings:
                szCol = headings[szCol]

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
    if isinstance(value, int) or isinstance(value, float):  # for int and float values
        splittingFunction = lambda row: row[column] >= value
    else:  # for strings
        splittingFunction = lambda row: row[column] == value
    list1 = [row for row in rows if splittingFunction(row)]
    list2 = [row for row in rows if not splittingFunction(row)]
    return (list1, list2)


def uniqueCounts(rows):
    """Counts the occurrences of each class in the dataset."""
    # The response variable is in the last column
    return Counter(row[-1] for row in rows)


def entropy(rows):
    if not rows:
        return 0.0
    total = len(rows)
    return -sum((count / total) * log2(count / total) for count in uniqueCounts(rows).values())


def gini(rows):
    """Calculates the Gini impurity for a list of rows.
    https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    if not rows:
        return 0.0
    total = len(rows)
    return 1.0 - sum((count / total) ** 2 for count in uniqueCounts(rows).values())


# def variance(rows):
#     if not rows:
#         return 0.0
#     total = len(rows)
#     data = [row[-1] for row in rows]
#     mean = sum(data) / total
#     variance = sum([(d - mean) ** 2 for d in data]) / total
#     return variance


def growDecisionTreeFrom(rows, evaluationFunction=entropy):
    """Grows and then returns a binary decision tree.
    evaluationFunction: entropy or gini"""

    if len(rows) == 0:
        return DecisionNode()
    currentScore = evaluationFunction(rows)

    bestGain = 0.0
    bestAttribute = None
    bestSets = None

    columnCount = len(rows[0]) - 1  # last column is the result/target column
    for col in range(columnCount):
        columnValues = set([row[col] for row in rows])

        for value in columnValues:
            (set1, set2) = divideSet(rows, col, value)

            # Gain -- Entropy or Gini
            p = len(set1) / len(rows)
            gain = (
                currentScore
                - p * evaluationFunction(set1)
                - (1 - p) * evaluationFunction(set2)
            )
            if gain > bestGain and len(set1) > 0 and len(set2) > 0:
                bestGain = gain
                bestAttribute = (col, value)
                bestSets = (set1, set2)

    if bestGain > 0:
        trueBranch = growDecisionTreeFrom(bestSets[0], evaluationFunction)
        falseBranch = growDecisionTreeFrom(bestSets[1], evaluationFunction)
        return DecisionNode(
            col=bestAttribute[0],
            value=bestAttribute[1],
            trueBranch=trueBranch,
            falseBranch=falseBranch,
        )
    else:
        return DecisionNode(results=uniqueCounts(rows))


def prune(tree, minGain, evaluationFunction=entropy, notify=False):
    """Prunes the obtained tree according to the minimal gain (entropy or Gini)."""
    # recursive call for each branch
    if tree.trueBranch.results == None:
        prune(tree.trueBranch, minGain, evaluationFunction, notify)
    if tree.falseBranch.results == None:
        prune(tree.falseBranch, minGain, evaluationFunction, notify)

    # merge leaves (potentionally)
    if tree.trueBranch.results != None and tree.falseBranch.results != None:
        tb, fb = [], []

        for v, c in tree.trueBranch.results.items():
            tb += [[v]] * c
        for v, c in tree.falseBranch.results.items():
            fb += [[v]] * c

        p = len(tb) / len(tb + fb)
        delta = (
            evaluationFunction(tb + fb)
            - p * evaluationFunction(tb)
            - (1 - p) * evaluationFunction(fb)
        )
        if delta < minGain:
            if notify:
                print("A branch was pruned: gain = %f" % delta)
            tree.trueBranch, tree.falseBranch = None, None
            tree.results = uniqueCounts(tb + fb)


def classify(observations, tree, dataMissing=False):
    """Classifies the observationss according to the tree.
    dataMissing: true or false if data are missing or not."""

    def classifyWithoutMissingData(observations, tree):
        if tree.results != None:  # leaf
            return tree.results
        else:
            v = observations[tree.col]
            branch = None
            if isinstance(v, int) or isinstance(v, float):
                if v >= tree.value:
                    branch = tree.trueBranch
                else:
                    branch = tree.falseBranch
            else:
                if v == tree.value:
                    branch = tree.trueBranch
                else:
                    branch = tree.falseBranch
        return classifyWithoutMissingData(observations, branch)

    def classifyWithMissingData(observations, tree):
        if tree.results != None:  # leaf
            return tree.results
        else:
            v = observations[tree.col]
            if v == None:
                tr = classifyWithMissingData(observations, tree.trueBranch)
                fr = classifyWithMissingData(observations, tree.falseBranch)
                tcount = sum(tr.values())
                fcount = sum(fr.values())
                tw = tcount / (tcount + fcount)
                fw = fcount / (tcount + fcount)
                keys = tr.keys() | fr.keys()
                return Counter({k: tr.get(k, 0) * tw + fr.get(k, 0) * fw for k in keys})
            else:
                branch = None
                if isinstance(v, int) or isinstance(v, float):
                    if v >= tree.value:
                        branch = tree.trueBranch
                    else:
                        branch = tree.falseBranch
                else:
                    if v == tree.value:
                        branch = tree.trueBranch
                    else:
                        branch = tree.falseBranch
            return classifyWithMissingData(observations, branch)

    # function body
    if dataMissing:
        return classifyWithMissingData(observations, tree)
    else:
        return classifyWithoutMissingData(observations, tree)

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
    """Loads a CSV file and converts all floats and ints into basic datatypes."""

    def convertTypes(s):
        s = s.strip()
        try:
            return float(s) if "." in s else int(s)
        except ValueError:
            return s

    with open(file, "rt") as f:
        reader = csv.reader(f)
        return [[convertTypes(item) for item in row] for row in reader]


def export_tree(tree, dot=None):
    """Recursively export a decision tree to Graphviz DOT format."""
    if dot is None:
        dot = Digraph()

    if tree.results is not None:
        # Leaf node
        label = ", ".join(f"{k}:{v}" for k, v in tree.results.items())
        dot.node(str(id(tree)), label, shape="box", style="filled", color="lightgrey")
    else:
        # Decision node
        if isinstance(tree.value, (int, float)):
            label = f"Column {tree.col} >= {tree.value}"
        else:
            label = f"Column {tree.col} == {tree.value}"
        dot.node(str(id(tree)), label, shape="ellipse")

        # Recurse
        dot.edge(str(id(tree)), str(id(tree.trueBranch)), label="yes")
        export_tree(tree.trueBranch, dot)
        dot.edge(str(id(tree)), str(id(tree.falseBranch)), label="no")
        export_tree(tree.falseBranch, dot)

    return dot


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
        help="Evaluation function to use for splits (default: entropy)",
    )
    parser.add_argument(
        "--plot",
        metavar="FILE",
        help="Export the decision tree as a Graphviz image (e.g. tree.png or tree.svg)",
    )
    args = parser.parse_args()

    eval_fn = entropy if args.criterion == "entropy" else gini

    # All examples do the following steps:
    #     1. Load training data
    #     2. Let the decision tree grow
    #     4. Plot the decision tree
    #     5. classify without missing data
    #     6. Classifiy with missing data
    #     (7.) Prune the decision tree according to a minimal gain level
    #     (8.) Plot the pruned tree

    decisionTree=None
    if args.example == 1:
        # the smaller example
        trainingData = loadCSV(
            "tbc.csv"
        )  # sorry for not translating the TBC and pneumonia symptoms
        decisionTree = growDecisionTreeFrom(trainingData, evaluationFunction=eval_fn)
        print(decisionTree)

        print("\n--- Classification Examples ---")

        # Example 1: A sample with complete data
        complete_sample = ["ohne", "leicht", "Streifen", "normal", "normal"]
        result1 = classify(complete_sample, decisionTree, dataMissing=False)
        print_classification_result(complete_sample, result1)

        missing_sample = [None, "leicht", None, "Flocken", "fiepend"]
        result1 = classify(missing_sample, decisionTree, dataMissing=True) 
        print_classification_result(missing_sample, result2)  # no longer unique
        # Don't forget if you compare the resulting tree with the tree in my presentation: here it is a binary tree!
    else:
        # the bigger example
        trainingData = loadCSV("fishiris.csv")  # demo data from matlab
        decisionTree = growDecisionTreeFrom(trainingData)
        print(decisionTree)

        # notify, when a branch is pruned (one time in this example)
        prune(decisionTree, 0.5, evaluationFunction=eval_fn, notify=True)
        print(decisionTree)

        print("\n--- Classification Examples ---")

        # Example 1: A sample with complete data
        complete_sample = [6.0, 2.2, 5.0, 1.5]
        result1 = classify(complete_sample, decisionTree, dataMissing=False)
        print_classification_result(complete_sample, result1)
        
        # Example 2: A sample with missing data
        missing_sample = [None, None, None, 1.5]
        result2 = classify(missing_sample, decisionTree, dataMissing=True)
        print_classification_result(missing_sample, result2) # no longer unique

    if args.plot:
        # Remove extension if present
        filename = args.plot.rsplit(".", 1)[0]
        fileformat = args.plot.rsplit(".", 1)[-1]
        dot = export_tree(decisionTree)
        dot.render(filename, format=fileformat, cleanup=True)
        print(f"Decision tree exported to {filename}.{fileformat}")
