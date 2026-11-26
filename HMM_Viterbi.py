from Bio import SeqIO
import math
import time
import tracemalloc

# Start measuring memory
tracemalloc.start()

# Start measuring runtime
start_time = time.perf_counter()

# Read the genome FASTA file
fasta_file = "GCA_000005845.2_ASM584v2_genomic.fna"
record = SeqIO.read(fasta_file, "fasta")

sequence = str(record.seq).upper()

# Initially label every nucleotide as non-coding
labels = ["N"] * len(sequence)

# Read the GFF annotation file
gff_file = "genomic.gff"

with open(gff_file, "r") as file:
    for line in file:

        if line.startswith("#"):
            continue

        columns = line.strip().split("\t")

        # Only use protein-coding sequences
        if columns[2] != "CDS":
            continue

        # Make sure annotation belongs to this sequence
        if columns[0] != record.id:
            continue

        # Convert GFF coordinates to Python indexes
        start = int(columns[3]) - 1
        end = int(columns[4])

        # Label the coding region
        for i in range(start, end):
            labels[i] = "C"

def calculate_initial_probabilities(labels):

    initial_counts = {
        "C": 0,
        "N": 0
    }

    # Use the annotated first nucleotide as the initial state
    initial_counts[labels[0]] += 1

    total = sum(initial_counts.values())

    initial_probabilities = {
        state: initial_counts[state] / total
        for state in ["C", "N"]
    }

    return initial_counts, initial_probabilities

def calculate_transition_probabilities(labels):
    states = ["C", "N"]

    # Initialize transition counts
    counts = {
        "C": {"C": 0, "N": 0},
        "N": {"C": 0, "N": 0}
    }

    # Count transitions between consecutive labels
    for i in range(len(labels) - 1):
        current_state = labels[i]
        next_state = labels[i + 1]
        counts[current_state][next_state] += 1

    # Calculate probabilities
    transition_matrix = {}

    for state in states:
        total = sum(counts[state].values())
        transition_matrix[state] = {}

        for next_state in states:
            transition_matrix[state][next_state] = (
                counts[state][next_state]
            ) / total

    return counts, transition_matrix

def calculate_emission_probabilities(sequence, labels):
    states = ["C", "N"]
    nucleotides = ["A", "T", "G", "C"]

    # Initialize emission counts
    counts = {
        "C": {"A": 0, "T": 0, "G": 0, "C": 0},
        "N": {"A": 0, "T": 0, "G": 0, "C": 0}
    }

    # Count nucleotides for each hidden state
    for nucleotide, state in zip(sequence.upper(), labels):

        if nucleotide in nucleotides:
            counts[state][nucleotide] += 1

    # Calculate probabilities
    emission_matrix = {}

    for state in states:
        total = sum(counts[state].values())
        emission_matrix[state] = {}

        for nucleotide in nucleotides:
            emission_matrix[state][nucleotide] = (
                counts[state][nucleotide]
            ) / total

    return counts, emission_matrix

# Calculate initial probabilities
initial_counts, initial_probabilities = (
    calculate_initial_probabilities(labels)
)

# Calculate transition probabilities
transition_counts, transition_matrix = (
    calculate_transition_probabilities(labels)
)

# Calculate emission probabilities
emission_counts, emission_matrix = (
    calculate_emission_probabilities(sequence, labels)
)


def viterbi(sequence, transition_matrix, emission_matrix, initial_probabilities):

    states = ["C", "N"]
    nucleotides = ["A", "T", "G", "C"]
    length = len(sequence)

    def log_probability(probability):
        if probability == 0:
            return float("-inf")
        return math.log(probability)

    # Store previous probabilities
    previous = {}

    # Store backpointers using compact byte arrays
    backpointer_C = bytearray(length)
    backpointer_N = bytearray(length)

    # Initialize first nucleotide
    for state in states:

        emission = emission_matrix[state].get(sequence[0], 0)

        if sequence[0] not in nucleotides:
            emission = 1.0

        previous[state] = (
            log_probability(initial_probabilities[state])
            + log_probability(emission)
        )

    # Process remaining nucleotides
    for i in range(1, length):

        nucleotide = sequence[i]
        current = {}

        for state in states:

            if nucleotide in nucleotides:
                emission = emission_matrix[state][nucleotide]
            else:
                emission = 1.0

            emission_log = log_probability(emission)

            probability_C = (
                previous["C"]
                + log_probability(transition_matrix["C"][state])
                + emission_log
            )

            probability_N = (
                previous["N"]
                + log_probability(transition_matrix["N"][state])
                + emission_log
            )

            if probability_C >= probability_N:
                current[state] = probability_C
                best_previous = 0
            else:
                current[state] = probability_N
                best_previous = 1

            if state == "C":
                backpointer_C[i] = best_previous
            else:
                backpointer_N[i] = best_previous

        previous = current

    # Determine the final state
    if previous["C"] >= previous["N"]:
        final_state = 0
    else:
        final_state = 1

    # Backtrack to recover the most likely sequence
    predicted_labels = [""] * length

    predicted_labels[length - 1] = states[final_state]

    for i in range(length - 1, 0, -1):

        if final_state == 0:
            final_state = backpointer_C[i]
        else:
            final_state = backpointer_N[i]

        predicted_labels[i - 1] = states[final_state]

    return predicted_labels

def calculate_metrics(actual_states, predicted_states):

    # Initialize counts
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    # Compare actual labels with predicted labels
    for actual, predicted in zip(actual_states, predicted_states):

        if actual == "C" and predicted == "C":
            true_positive += 1
        elif actual == "N" and predicted == "C":
            false_positive += 1
        elif actual == "C" and predicted == "N":
            false_negative += 1
        elif actual == "N" and predicted == "N":
            true_negative += 1

    # Calculate precision
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive > 0 else 0
    )

    # Calculate recall
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0 else 0
    )

    # Calculate accuracy
    accuracy = (
        (true_positive + true_negative) /
        len(actual_states)
    )

    # Calculate F1-score
    f1_score = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0
    )

    return {
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "F1-score": f1_score
    }


# print("First 200 DNA nucleotides:", sequence[:200])
# print("First 200 states:", "".join(labels[:200]))

# # Display transition results
# print("\nTransition Counts:")
# for state, transitions in transition_counts.items():
#     print(state, transitions)

# print("\nTransition Probability Matrix:")
# for state, transitions in transition_matrix.items():
#     print(state, transitions)

# # Display emission results
# print("\nEmission Counts:")
# for state, emissions in emission_counts.items():
#     print(state, emissions)

# print("\nEmission Probability Matrix:")
# for state, emissions in emission_matrix.items():
#     print(state, emissions)

predicted_labels = viterbi(
    sequence,
    transition_matrix,
    emission_matrix,
    initial_probabilities
)

# Calculate performance metrics
metrics = calculate_metrics(labels, predicted_labels)

# # Display results
# print("\nViterbi Results:")
# print("Actual states:   ", "".join(labels[:20000]))
# print("Predicted states:", "".join(predicted_labels[:20000]))

print("\nPerformance Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# Stop measuring runtime
end_time = time.perf_counter()

# Get memory usage
current_memory, peak_memory = tracemalloc.get_traced_memory()

# Stop memory tracking
tracemalloc.stop()

# Display results
runtime = end_time - start_time

print(f"\nRuntime: {runtime:.4f} seconds")
print(f"Current memory: {current_memory / (1024 ** 2):.2f} MB")
print(f"Peak memory: {peak_memory / (1024 ** 2):.2f} MB")