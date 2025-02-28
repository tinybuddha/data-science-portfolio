# --------------------------------------------------------------------------------------------------------
#
# 6CCS3ML1 - Coursework 1, Niki Norgren, niki.norgren@kcl.ac.uk
# 17th February 2025
#
# Code references: inspiration for the implementation of a decision tree algorithm taken from Chapter 3.4, 
# pages 55-59, Machine Learning, Tom M. Mitchell, McGraw Hill, 1997. 
# Further, week 2 lecture material provided insights on decision trees as well as how to implement the 
# logic for entropy and information gain.
# 
# --------------------------------------------------------------------------------------------------------

# An implementation of a decision tree, upon which the actions of Pacman are chosen.


import math
import random
from collections import Counter, defaultdict


class Classifier:
    def __init__(self):
        self.tree = None 

    def reset(self):
        pass

    @staticmethod
    def calculate_entropy(target):
        # Use the Counter class to count the occurrences of each class in the target list
        class_counts = Counter(target)
        
        # Calculate the probability of each class by dividing the count of each class
        # by the total number of instances in the target list
        class_probabilities = [count / len(target) for count in class_counts.values()]
        
        # Calculate the entropy using the formula:
        # Entropy = -sum(p * log2(p)) for all probabilities p where p > 0
        # The condition `if p > 0` prevents math errors due to log(0)
        return -sum(p * math.log2(p) for p in class_probabilities if p > 0)


    @staticmethod
    def calculate_information_gain(data, split_feature, target):
        # Calculate the total entropy of the target data before any splits
        original_entropy = Classifier.calculate_entropy(target)

        # Create a list of the values from the dataset for the feature we are considering for a split
        feature_values = [row[split_feature] for row in data]

        # A dictionary to hold subsets of target values corresponding to each unique feature value
        target_subsets = defaultdict(list)
        for feature_value, target_value in zip(feature_values, target):
            # Append the target value to the list in the dictionary key of the feature value
            target_subsets[feature_value].append(target_value)
        
        # Total number of instances in the data
        total_instances = len(data)

        # Calculate the weighted entropy after the split:
        # for each subset of target values, calculate its entropy and multiply by its proportion of the total data
        weighted_entropy = sum((len(subset) / total_instances) * Classifier.calculate_entropy(subset) for subset in target_subsets.values())

        # Information gain is the reduction in entropy due to the split, so subtract the weighted entropy from the original entropy
        return original_entropy - weighted_entropy


    # Here we recursively build the decision tree
    def decision_tree(self, data, target, features):
        # Base case: if all target values are the same, return a leaf node with that value
        if len(set(target)) == 1:
            return DecisionNode(question=target[0])
        
        # Base case: if no features are left to split on, return a leaf node with the most common target value
        if not features:
            common_target = Counter(target).most_common(1)[0][0]
            return DecisionNode(question=common_target)

        # Calculate information gain for each feature and find the best one to split on
        gains = [self.calculate_information_gain(data, feature, target) for feature in features]
        best_feature = features[gains.index(max(gains))]

        # Create a new decision node for the best feature with placeholders for its branches
        node = DecisionNode(question=f"Feature {best_feature}")
        
        # Get all unique values of the best feature to determine how to split the data
        feature_values = set(row[best_feature] for row in data)

        # Remove the best feature from the list of features since it will be used for the current split
        new_features = features[:]
        new_features.remove(best_feature)

        # For each unique value of the best feature, partition the data and recursively build subtrees
        for value in feature_values:
            # Indices of all data points where the best feature has this value
            indices = [i for i, row in enumerate(data) if row[best_feature] == value]
            # Subset of data and target for each value
            branches_data = [data[i] for i in indices]
            branches_target = [target[i] for i in indices]
            
            # Recursively build the tree for each branch and assign it to the corresponding branch in the node
            node.branches[value] = self.decision_tree(branches_data, branches_target, new_features)

        return node 
    
    
   # Fit the classifier with the provided data and target
    def fit(self, data, target):
        # Generate a list of feature indices based on the number of features in a sample
        features = list(range(len(data[0])))  

        # Build the decision tree using the recursive decision_tree method. This method will use the
        # features and target data to construct the tree, deciding the best feature to split at each node
        # based on the calculated information gain from the data provided
        self.tree = self.decision_tree(data, target, features)


    # We predict our action for Pacman
    def predict(self, features, legal):
        # Start at the root of the decision tree
        node = self.tree

        # Traverse the tree based on the current feature values
        while isinstance(node, DecisionNode) and node.branches:
            # Extract the feature index from the node's question and get the corresponding feature value
            value = features[int(node.question.split()[1])]

            # Move to the next node in the path based on the feature value
            node = node.branches.get(value, None)

            # If no such branch exists, break the loop (no further traversal possible)
            if node is None:
                break

        # Determine the action to return:
        # - If a valid leaf node is found, use its question as the predicted action
        # - If no valid node is found or the path breaks, choose a random legal action
        predicted_action = node.question if node else random.choice(legal)

        # Return the predicted action if it's in the list of legal actions, otherwise choose randomly from legal actions
        # This ensures that the action returned is always valid and executable in the game state
        return predicted_action if predicted_action in legal else random.choice(legal)


# Represents a decision node or leaf in the decision tree
class DecisionNode:
    def __init__(self, question=None, branches=None):
        self.question = question  # string describing the question
        self.branches = branches or {}  # dict of value: DecisionNode