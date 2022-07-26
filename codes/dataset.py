from remove_anno import run_remove_annotations
from graph_extraction import run_extraction
from node_numbering import run_graph_numbering
from process_dataset import process_dataset

def main():
    # path to original dataset
    # SHOULD BE MODIFIED!
    dataset_path = "../data_add/"

    # remove annotations
    #run_remove_annotations(dataset_path)

    # extract graphs
    #run_extraction("../removed_anno/")
    run_extraction(dataset_path)

    # re-numbering nodes in the graph
    run_graph_numbering("../complete_add/")

    # process dataset
    process_dataset("../complete_add/")

    return
    

if __name__ == "__main__":
    main()