import argparse
from ast import parse
import json
import logging
import textwrap
from typing import (
    List,
    Optional,
)

from pfd.utils.download_pfd_artifacts import (
    print_op_download_setting,
)
from pfd import __version__
from .download import download, download_by_def, download_end_result
from .status import status
from .submit import FlowGen, resubmit_workflow
from .common import (
    expand_idx,
    perturb_cli
)


def main_parser() -> argparse.ArgumentParser:
    """PFD-kit commandline options argument parser.

    Notes
    -----
    This function is used by documentation.

    Returns
    -------
    argparse.ArgumentParser
        the argument parser
    """
    parser = argparse.ArgumentParser(
        description="PFD-kit: fine-tune and distillation from pre-trained atomic models"
        "machine learning potential energy models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(title="Valid subcommands", dest="command")

    ##########################################
    # submit
    parser_run = subparsers.add_parser(
        "submit",
        help="Submit workflows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_run.add_argument(
        "CONFIG", help="the config file in json format defining the workflow."
    )
    parser_run.add_argument(
        "-m",
        "--monitering",
        action="store_false",
        help="Keep monitering the progress",
    )

    ##########################################
    # resubmit
    parser_resubmit = subparsers.add_parser(
        "resubmit",
        help="Submit workflows",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_resubmit.add_argument(
        "CONFIG", help="the config file in json format defining the workflow."
    )
    parser_resubmit.add_argument("ID", help="the ID of existing workflow")
    parser_resubmit.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="list the Steps of the existing workflow.",
    )
    parser_resubmit.add_argument(
        "-u",
        "--reuse",
        type=str,
        nargs="+",
        default=None,
        help="specify which Steps to reuse. e.g., 0-41,\
            the first to the 41st steps.",
    )
    parser_resubmit.add_argument(
        "-f",
        "--fold",
        action="store_true",
        help="if set then super OPs are folded to be reused in the new workflow",
    )
    parser_resubmit.add_argument(
        "-m",
        "--monitering",
        action="store_false",
        help="Keep monitering the progress",
    )
    parser_resubmit.add_argument(
        "--unsuccessful-step-keys",
        action="store_true",
        help="List and reuse all unsuccessful step keys from previous workflow",
    )
    ##########################################
    # download
    parser_download = subparsers.add_parser(
        "download",
        help=(
            "Download the artifacts of PFD workflow steps. User needs to provide the input json file as well as the workflow ID. The command would then download the end model if workflow is successfully completed.\n"
        ),
        description=(
            textwrap.dedent(
                """
            """
            )
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser_download.add_argument("CONFIG", help="the config file in json format.")
    parser_download.add_argument("ID", help="the ID of the existing workflow.")
    parser_download.add_argument(
        "-l",
        "--list-supported",
        action="store_true",
        help="list all supported steps artifacts",
    )
    parser_download.add_argument(
        "-k",
        "--keys",
        type=str,
        nargs="+",
        help="the keys of the downloaded steps. If not provided download all artifacts",
    )
    parser_download.add_argument(
        "-i",
        "--iterations",
        type=str,
        nargs="+",
        help="the iterations to be downloaded, support ranging expression as 0-10.",
    )
    parser_download.add_argument(
        "-d",
        "--step-definitions",
        type=str,
        nargs="+",
        help="the definition for downloading step artifacts",
    )
    parser_download.add_argument(
        "-p",
        "--prefix",
        type=str,
        help="the prefix of the path storing the download artifacts",
    )
    parser_download.add_argument(
        "-n",
        "--no-check-point",
        action="store_false",
        help="if specified, download regardless whether check points exist.",
    )
    #########################################
    # perturb
    parser_perturb = subparsers.add_parser(
        "perturb",
        help="Perturb structures from files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_perturb.add_argument(
        "ATOMS",
        type=str,
        nargs="+",
        help="the structure files to be perturbed, support multiple files.",
    )
    parser_perturb.add_argument(    
        "-n",
        "--pert-num",
        type=int,
        default=1,
        help="the number of perturbed structures to be generated for each input structure.",
    )
    parser_perturb.add_argument(
        "-c",
        "--cell-pert-fraction",
        type=float,
        default=0.05,
        help="the fraction of cell perturbation.",
    )   
    parser_perturb.add_argument(
        "-d",
        "--atom-pert-distance",
        type=float,
        default=0.2,
        help="the distance to perturb the atom.",
    )
    parser_perturb.add_argument(
        "-s",
        "--atom-pert-style",
        type=str,
        default="normal",
        help="the style of perturbation.",
    )
    parser_perturb.add_argument(    
        "-a",
        "--atom-pert-prob",
        type=float,
        default=1.0,
        help="the probability of perturbing each atom.",
    )
    parser_perturb.add_argument(    
        "-r",
        "--supercell",
        type=int,
        nargs="+",
        default=None,
        help="the supercell replication, support int or 3 ints.",
    )
    #########################################
    # status
    parser_status = subparsers.add_parser(
        "status",
        help="Check exploration status",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_status.add_argument(
        "CONFIG", help="the config file in json format defining the workflow."
    )
    parser_status.add_argument("ID", help="the ID of existing workflow")

    # Add the version argument
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="show the version number and exit",
    )
    return parser


def parse_args(args: Optional[List[str]] = None):
    """PFD-kit commandline options argument parsing.

    Parameters
    ----------
    args : List[str]
        list of command line arguments, main purpose is testing default option None
        takes arguments from sys.argv
    """
    parser = main_parser()

    parsed_args = parser.parse_args(args=args)
    if parsed_args.command is None:
        parser.print_help()

    return parsed_args


def main():
    # logging
    logging.basicConfig(level=logging.INFO)
    logo = r"""
    ____  _____ ____    _  __ _ _   
   |  _ \|  ___|  _ \  | |/ /(_) |_ 
   | |_) | |_  | | | | | / / | | __|
   |  __/|  _| | |_| | |  /\ | | |_ 
   |_|   |_|   |____/  |_|\_\|_|\__|
    """
    print(logo)
    args = parse_args()
    if args.command == "submit":
        print("Submitting workflow")
        with open(args.CONFIG) as fp:
            config = json.load(fp)
        FlowGen(config).submit(only_submit=args.monitering)

    elif args.command == "resubmit":
        with open(args.CONFIG) as fp:
            config = json.load(fp)
        wfid = args.ID
        resubmit_workflow(
            wf_config=config,
            wfid=wfid,
            list_steps=args.list,
            reuse=args.reuse,
            fold=args.fold,
            only_submit=args.monitering,
            unsuccessful_step_keys=args.unsuccessful_step_keys
        )

    elif args.command == "download":
        with open(args.CONFIG) as fp:
            config = json.load(fp)
        wfid = args.ID
        if args.list_supported is not None and args.list_supported:
            print(print_op_download_setting())
        elif args.keys is not None:
            download(
                wfid,
                config,
                wf_keys=args.keys,
                prefix=args.prefix,
                chk_pnt=args.no_check_point,
            )
        elif args.step_definitions:
            download_by_def(
                wfid,
                config,
                iterations=(
                    expand_idx(args.iterations) if args.iterations is not None else None
                ),
                step_defs=args.step_definitions,
                prefix=args.prefix,
                chk_pnt=args.no_check_point,
            )
        else:
            download_end_result(wfid, config, prefix=args.prefix)
    elif args.command == "status":
        with open(args.CONFIG) as fp:
            config = json.load(fp)
        wfid = args.ID
        status(wfid, config)
    elif args.command == "perturb":
        perturb_cli(
            atoms_path_ls=args.ATOMS,
            pert_num=args.pert_num,
            cell_pert_fraction=args.cell_pert_fraction,
            atom_pert_distance=args.atom_pert_distance,
            atom_pert_style=args.atom_pert_style,
            atom_pert_prob=args.atom_pert_prob,
            supercell=args.supercell
        )
    elif args.command is None:
        pass
    else:
        raise RuntimeError(f"unknown command {args.command}")
