import sys
import os.path


###############################################################################
RANDOM_SEED = 9879

ALLOWED_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PREFERRED_LENGTH_RANGE = [5, 10]

MIN_INITIAL_PEPTIDE_LENGTH = 3
MAX_INITIAL_PEPTIDE_LENGTH = 24
POPULATION_SIZE = 50
OFFSPRING_COUNT = 30
MAX_NUM_GENERATIONS = 30
TOURNAMENT_SIZE = 3
MUTATION_PROBABILITY = 0.05
###############################################################################


if len(sys.argv) < 3 or sys.argv[1] not in ["--ml-model", "--ml-model-path"]:
    raise Exception("ML model not specified.")
elif sys.argv[1] == "--ml-model" and sys.argv[2] not in ["AP", "AP_SP", "SP", "TSNE_AP_SP", "TSNE_SP"]:
    raise Exception(f"There is no pre-trained model named {sys.argv[2]}.")
elif sys.argv[1] == "--ml-model-path" and not os.path.exists(sys.argv[2]):
    raise Exception(f"The model at {sys.argv[2]} does not exist.")
elif sys.argv[1] == "--ml-model" and not os.path.exists(f"./SA_ML_predictive/models/{sys.argv[2]}.h5"):
    raise Exception(f"Can't find the pre-trained model in the 'SA_ML_predictive/models' folder. "
                    f"Try running the code from the project's root directory ('ml_peptide_self_assembly').")


sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+"/SA_ML_predictive/code/")


import tensorflow as tf
import numpy as np
from genetic_algorithm_library import GeneticAlgorithm
from SA_ML_predictive.code import utils
from SA_ML_predictive.code import automate_training


np.random.seed(RANDOM_SEED)
MODEL_PATH = ""

if sys.argv[1] == "--ml-model":
    MODEL_PATH = f"./SA_ML_predictive/models/{sys.argv[2]}.h5"
elif sys.argv[1] == "--ml-model-path":
    MODEL_PATH = sys.argv[2]

SA_ML_MODEL = tf.keras.models.load_model(MODEL_PATH)


def predict_SA_probability(sequence):
    if len(sequence) > 24:
        raise Exception("Peptide is too long.")
    elif len(sequence) < 1:
        raise Exception("No peptide for prediction.")

    pep_list = [sequence] 

    seq_example = ""
    for _ in range(utils.MAX_LEN):
        seq_example += "A"
    pep_list.append(seq_example)
    pep_labels = ["1", "1"]
  
    offset = 1

    properties = np.ones(95)
    properties[0] = 0
    mask_value = 2

    model_type = MODEL_PATH.split("/")[-1].replace(".h5", "")
  
    SA, NSA = utils.load_data(model_type, [pep_list, pep_labels], offset, properties, mask_value)
    all_data, all_labels = utils.merge_data(SA, NSA)
      
    test_data, test_labels = utils.reshape_for_model(model_type, all_data, all_labels)
    model_predictions = SA_ML_MODEL.predict(test_data, batch_size=automate_training.BATCH_SIZE)
    model_predictions = utils.convert_list(model_predictions)

    return model_predictions[0]


def calculate_amino_acid_frequencies(sequence):
    frequencies = []

    for amino_acid in ALLOWED_AMINO_ACIDS:
        frequencies.append(sequence.count(amino_acid))

    return np.array(frequencies)


def calculate_similarity_penalty(sequence, population):
    penalty = 0
    first = True

    frequencies = calculate_amino_acid_frequencies(sequence)

    for neighbour_peptide in population:
        if neighbour_peptide.sequence == sequence and first:
            first = False
            continue

        neighbour_frequencies = calculate_amino_acid_frequencies(neighbour_peptide.sequence)
        penalty += 0.1 * (1 - np.sum(np.abs(frequencies - neighbour_frequencies))
                          / (np.sum(frequencies) + np.sum(neighbour_frequencies)))

    return penalty / (len(population) - 1)


def calculate_length_penalty(sequence):
    if len(sequence) < PREFERRED_LENGTH_RANGE[0] or len(sequence) > PREFERRED_LENGTH_RANGE[1]:
        return min(0.05 * abs(len(sequence) - (PREFERRED_LENGTH_RANGE[0] + PREFERRED_LENGTH_RANGE[1]) / 2), 0.5)
    return 0


###############################################################################


GA = GeneticAlgorithm(
    fitness_function=predict_SA_probability,
    similarity_penalty=calculate_similarity_penalty,
    length_penalty=calculate_length_penalty,
    min_initial_peptide_length=MIN_INITIAL_PEPTIDE_LENGTH,
    max_initial_peptide_length=MAX_INITIAL_PEPTIDE_LENGTH,
    allowed_amino_acids=ALLOWED_AMINO_ACIDS,
    population_size=POPULATION_SIZE,
    offspring_count=OFFSPRING_COUNT,
    max_num_generations=MAX_NUM_GENERATIONS,
    tournament_size=TOURNAMENT_SIZE,
    mutation_probability=MUTATION_PROBABILITY
)

final_population = GA.find_peptides()

best_sequences = list(set([peptide.sequence for peptide in final_population]))
suggested_sequences = []

for sequence in best_sequences:
    SA_probability = predict_SA_probability(sequence)
    suggested_sequences.append([sequence, SA_probability])

suggested_sequences = sorted(suggested_sequences, key=lambda list_member: list_member[1], reverse=True)

print_list = ["Peptide,Self-assembly probability [%]\n"]
print_list += [
    f"{sequence},{np.round(SA_probability * 100, decimals=1)}%\n"
    for sequence, SA_probability in suggested_sequences
]

with open(f"suggested_SA_peptides_{PREFERRED_LENGTH_RANGE[0]}_{PREFERRED_LENGTH_RANGE[1]}.csv", "w") as file:
    file.writelines(print_list)
