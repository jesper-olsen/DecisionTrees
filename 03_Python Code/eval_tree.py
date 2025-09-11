import random
import argparse
from statistics import mean, stdev  # Import mean and stdev for final summary
from decision_tree import DecisionTree, load_csv


def create_folds(data, k):
    """
    Splits data into k folds for cross-validation.

    Args:
        data (list of lists): The full dataset.
        k (int): The number of folds to create.

    Returns:
        list of lists: A list where each element is a fold (a list of data rows).
    """
    # Shuffle the data to ensure randomness before creating folds
    shuffled_data = data[:]
    random.seed(42)  # Use a seed for reproducible results
    random.shuffle(shuffled_data)

    folds = []
    fold_size = len(shuffled_data) // k
    start_index = 0
    for i in range(k):
        # The last fold gets all the remaining data points
        end_index = start_index + fold_size if i < k - 1 else len(shuffled_data)
        folds.append(shuffled_data[start_index:end_index])
        start_index = end_index

    return folds


def split_data(data, split_ratio=0.8):
    """
    Splits a dataset into training and testing sets.
    """
    shuffled_data = data[:]
    random.seed(42)
    random.shuffle(shuffled_data)
    split_index = int(len(shuffled_data) * split_ratio)
    train_data = shuffled_data[:split_index]
    test_data = shuffled_data[split_index:]
    return train_data, test_data


def calculate_accuracy(model, test_data):
    """
    Calculates the accuracy of a trained model on a test set.
    """
    if not test_data:
        return 0.0
    correct_predictions = 0
    total_predictions = len(test_data)
    for row in test_data:
        features = row[:-1]
        true_label = row[-1]
        result_dict = model.classify(features, handle_missing=False)
        if result_dict:
            predicted_label = max(result_dict, key=result_dict.get)
            if predicted_label == true_label:
                correct_predictions += 1
    accuracy = (correct_predictions / total_predictions) * 100
    return accuracy


def kfold_eval(data, args):
    # --- K-FOLD CROSS-VALIDATION LOGIC ---
    print(f"\nPerforming {args.k_folds}-fold cross-validation...")
    folds = create_folds(data, args.k_folds)
    fold_accuracies = []

    for i in range(args.k_folds):
        # The i-th fold is the test set
        test_data = folds[i]
        # All other folds combined are the training set
        train_folds = folds[:i] + folds[i + 1 :]
        train_data = [
            row for fold in train_folds for row in fold
        ]  # Flatten list of lists

        print(f"\n--- Fold {i+1}/{args.k_folds} ---")
        print(
            f"Training on {len(train_data)} samples, testing on {len(test_data)} samples."
        )

        # Train the model for this fold
        model = DecisionTree.train(train_data, header, criterion=args.criterion, max_depth=args.max_depth, min_samples_split=args.min_samples_split)
        print(f"Trained a model with {model.size()} nodes")

        # Prune if requested
        if args.prune > 0.0:
            model.prune(min_gain=args.prune, criterion=args.criterion)
            print(f"Pruned model down to {model.size()} nodes")

        # Evaluate and store accuracy
        accuracy = calculate_accuracy(model, test_data)
        fold_accuracies.append(accuracy)
        print(f"Fold {i+1} Accuracy: {accuracy:.2f}%")

    # After the loop, print the final summary
    print("\n" + "=" * 30)
    print("Cross-Validation Summary")
    print("=" * 30)
    avg_accuracy = mean(fold_accuracies)
    std_dev = stdev(fold_accuracies) if len(fold_accuracies) > 1 else 0.0
    print(f"Average Accuracy: {avg_accuracy:.2f}%")
    print(f"Standard Deviation: {std_dev:.2f}%")
    print("=" * 30)


def split_eval(data, args):
    # --- SIMPLE TRAIN/TEST SPLIT LOGIC ---
    print("\nPerforming a simple train/test split...")
    train_data, test_data = split_data(data, split_ratio=args.split_ratio)
    print(
        f"Data split into {len(train_data)} training samples and {len(test_data)} test samples."
    )
    print("-" * 30)

    print(f"Training the decision tree model (criterion: {args.criterion})...")
    model = DecisionTree.train(train_data, header, criterion=args.criterion, max_depth=args.max_depth, min_samples_split=args.min_samples_split)
    print(f"Trained a model with {model.size()} nodes")

    if args.prune > 0.0:
        print(f"Pruning the tree with min_gain = {args.prune}...")
        model.prune(min_gain=args.prune, criterion=args.criterion, notify=True)
        print(f"Pruned down to {model.size()} nodes")
    else:
        print("No pruning requested.")

    if args.plot:
        model.export_graph(args.plot)

    print("\nEvaluating model accuracy on the test set...")
    accuracy = calculate_accuracy(model, test_data)

    print("\n--- Evaluation Result ---")
    print(f"Model Accuracy: {accuracy:.2f}%")
    print("-" * 30)


if __name__ == "__main__":
    # 1. Set up the Argument Parser
    # 
    parser = argparse.ArgumentParser(
        description="Train and evaluate a decision tree on a given dataset."
    )
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the tree. Default: None (unlimited)")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Minimum number of samples required to split a node (default: 2)")
    parser.add_argument("file_path", type=str, help="Path to the training CSV dataset file.")
    parser.add_argument(
        "--test_file",
        type=str,
        help="Optional path to a separate CSV test dataset file. If provided, overrides split/k-fold.",
    )
    parser.add_argument(
        "-s",
        "--split_ratio",
        type=float,
        default=0.8,
        help="Proportion for training in a simple split. Default: 0.8",
    )
    parser.add_argument(
        "-c",
        "--criterion",
        type=str,
        choices=["entropy", "gini"],
        default="gini",
        help="The splitting criterion to use. Default: 'gini'",
    )
    parser.add_argument(
        "-p",
        "--prune",
        type=float,
        default=0.0,
        help="Minimum gain to keep a branch. Default: 0.0 (no pruning).",
    )
    parser.add_argument(
        "--plot",
        metavar="FILE",
        help="Export the decision tree as a Graphviz image (e.g. tree.png or tree.svg)",
    )
    parser.add_argument(
        "-k",
        "--k_folds",
        type=int,
        default=0,
        help="Number of folds for k-fold cross-validation. If > 1, this overrides --split_ratio. Default: 0",
    )

    args = parser.parse_args()
    if args.test_file:
        print(f"Loading training dataset from: {args.file_path}...")
        header, train_data = load_csv(args.file_path)
        print(f"Training set: {len(train_data)} rows")

        print(f"Loading test dataset from: {args.test_file}...")
        test_header, test_data = load_csv(args.test_file)
        print(f"Test set: {len(test_data)} rows")

        # train
        print(f"\nTraining decision tree (criterion: {args.criterion})...")
        model = DecisionTree.train(train_data, header, criterion=args.criterion, max_depth=args.max_depth, min_samples_split=args.min_samples_split)
        print(f"Trained a model with {model.size()} nodes")

        if args.prune > 0.0:
            print(f"Pruning the tree with min_gain = {args.prune}...")
            model.prune(min_gain=args.prune, criterion=args.criterion, notify=True)
            print("Pruning complete.")

        if args.plot:
            model.export_graph(args.plot)

        print("\nEvaluating model accuracy on the external test set...")
        accuracy = calculate_accuracy(model, test_data)
        print(f"\n--- Evaluation Result ---")
        print(f"Model Accuracy: {accuracy:.2f}%")
        print("-" * 30)
    else:
        # --- Case 2: single file, split or k-fold ---
        # 2. Load the CSV dataset
        print(f"Loading dataset from: {args.file_path}...")
        try:
            header, data = load_csv(args.file_path)
        except FileNotFoundError:
            print(f"Error: The file '{args.file_path}' was not found.")
            exit()
        print(f"Dataset loaded successfully with {len(data)} rows.")

        # 3. Choose evaluation method: Cross-Validation or Simple Split
        if args.k_folds > 1:
            kfold_eval(data, args)
        else:
            split_eval(data, args)
