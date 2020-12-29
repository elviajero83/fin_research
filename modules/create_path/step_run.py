"""
 merge input datasets
"""

import argparse
import logging
import os, sys
import pandas as pd
from shutil import copyfile
from os.path import dirname, abspath
import pandas as pd


from utils.args_utils import str2bool, args_dict_creator
from utils.log_utils import log, DataCategory, logger_init, aml_run_attach
from utils.file_utils import select_files_from_paths

# from pathlib import Path

# sys.path.append(abspath(Path(__file__).parents[2]))
# from kuri.utils.file_utils import (
#     get_normalized_output_file_path,
#     select_first_file_or_file,
#     unzip_compressed,
# )
# from kuri.utils import log, DataCategory
# from kuri.aml import aml_run_attach, log_directory, logger_init
# from kuri.aml.arg_utils import CompliantArgumentParser
# from kuri.aml.arg_utils import AMLCompatibleInputPath, AMLCompatibleOutputPath
# from kuri.utils.arg_utils import str2bool, args_dict_creator


def get_arg_parser(parser=None):
    """Adds module arguments to a given argument parser.

    Args:
        parser (argparse.ArgumentParser or CompliantArgumentParser): an argument parser instance

    Returns:
        CompliantArgumentParser: the argument parser instance

    Notes:
        if parser is None, creates a new parser instance
    """
    # add arguments that are specific to the module
    if parser is None:
        parser = argparse.ArgumentParser()
        # parser = CompliantArgumentParser(description=__doc__)

    # add arguments that are specific to the module
    group = parser.add_argument_group("Module arguments")
    group.add_argument(
        "--path_to_file",
        type=str,
        help="path on cosmos",
        required=True,
    )
    group.add_argument(
        "--file_path",
        type=str,
        help="file containing the path",
        required=True,
    )

    # add arguments for pipeline launch
    group = parser.add_argument_group("Run arguments [generic]")
    group.add_argument(
        "--verbose",
        dest="verbose",
        type=str2bool,
        required=False,
        default=True,
        help="increases verbosity of prints/logging.",
    )
    group.add_argument(
        "--log-in-aml",
        dest="log_in_aml",
        type=str2bool,
        required=False,
        default=True,
        help="send metrics to AzureML (default: True)",
    )

    return parser


def main(flags=None):
    """Main method for calling module.

    Args:
        flags (List[str], optional): list of flags to feed script, useful for debugging. Defaults to None.
    """

    parser = get_arg_parser()

    # screams an exception and sys.exit() if there's an error
    # sends a warning on logs if there's an unknown arg
    args, _ = parser.parse_known_args(flags)
    args_dict = args_dict_creator(args)

    if args.verbose:
        logger_init(logging.DEBUG)
    else:
        logger_init(logging.INFO)

    if args.log_in_aml:
        # imports azureml.core.run and get run from context
        aml_run_attach()

    input_path_args_list = []
    output_path_args_list = ["file_path"]
    files_dict = select_files_from_paths(
        args_dict, input_path_args_list, output_path_args_list
    )

    with open(files_dict["file_path"], "w") as f:
        f.write(args_dict["path_to_file"])


if __name__ == "__main__":
    main()
