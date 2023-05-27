from utils.graph_generator import update_sample_graph_data
import argparse


def update_yaml_data():
    parser = argparse.ArgumentParser(description='Updates the yaml cache files for opinion_formingui.py. This will replace any preexisting yaml files of the same vertices.')
    parser.add_argument('lower_vertex_value', type=int,
                        help='The lower bound vertex count you want to be saved in YAML cache files. Must be greater than 5')
    parser.add_argument('higher_vertex_value', type=int,
                        help='The upper bound vertex count you want to be saved in YAML cache files. Must be less than 17')
    args = parser.parse_args()
    if args.higher_vertex_value > 16:
        print('Error: higher vertex value must be less than 17')
    elif args.lower_vertex_value <= 5:
        print('Error: lower vertex value must be greater than 5')
    else:
        update_sample_graph_data(lower_vertices_count=args.lower_vertex_value, upper_vertices_count=args.higher_vertex_value)


if __name__ == "__main__":
    update_yaml_data()
