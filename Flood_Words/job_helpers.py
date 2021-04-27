""" Helper functions for processing jobs
"""


def construct_output_filename(input_path, output_path, parameters):
    """ Construct an output filename with input parameters

    :param input_path: file path for input
    :param output_path: directory path for output
    :param parameters: specific options to be included in output filename
    """

    # append slash to output directory if it doesn't exist
    if not output_path.endswith("/"):
        output = output_path + "/"
    else:
        output = output_path

    # extract file name after final slash in path, and remove unsupported chars
    if input_path:
        input_file = input_path.split("/")[-1] \
            .replace("_*", "") \
            .replace("*.json", "") \
            .replace("*.json.gz", "") \
            .replace("*", "") \
            .replace(".", "")

        # pre-pend input file to filename components
        parameters += [input_file]

    # filter parameters that do not have a value
    filename_components = [p for p in parameters if p or p == 0]

    # append the input parameters delimited by an underscore to the path
    output += "_".join([str(c) for c in filename_components])

    return output
