import os


def MINIMUM_VERTICES_SAMPLE_SIZE():
    return 6


def SAMPLE_GRAPH_COUNT():
    return 1000


def SAMPLE_BATCH_SIZE():
    return 100


def DATA_DIR():
    pardir = os.path.relpath(os.path.join("", "data"))
    return pardir


def SEQUENCED_GRAPH_COUNT_FILE_PATH():
    return os.path.relpath(os.path.join(DATA_DIR(), 'sequenced_graph_counts.yaml'))


def SAMPLED_MATRICES_FILE_PATH():
    return os.path.relpath(os.path.join(DATA_DIR(), 'sampled_matrices_v1.yaml'))


def GRAPH_DATA_CSV_FILE_PATH():
    return os.path.relpath(os.path.join(DATA_DIR(), 'all_graph_data_1.csv.gz'))
