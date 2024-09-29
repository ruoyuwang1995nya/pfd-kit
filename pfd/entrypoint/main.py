import argparse
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

from .download import (
    download,
    download_by_def,
)

from .submit import submit_dist, submit_ft, resubmit_workflow

from .common import (
    expand_idx,
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
        "-t",
        "--task",
        choices=["dist", "finetune"],
        help="Specify the task to be executed.",
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
        "-t",
        "--task",
        choices=["dist", "finetune"],
        help="Specify the task to be executed.",
    )
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

    ##########################################
    # download
    parser_download = subparsers.add_parser(
        "download",
        help=("Download the artifacts of PFD workflow steps.\n"),
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
    return parser


def parse_args(args: Optional[List[str]] = None):
    """DP-DISTILL commandline options argument parsing.

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

    args = parse_args()
    if args.command == "submit":
        print("Submitting workflow")
        if args.task == "dist":
            with open(args.CONFIG) as fp:
                config = json.load(fp)
            submit_dist(
                config,
            )
        elif args.task == "finetune":
            with open(args.CONFIG) as fp:
                config = json.load(fp)
            submit_ft(
                config,
            )
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
            flow_type=args.task,
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
        else:
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

    elif args.command is None:
        pass
    else:
        raise RuntimeError(f"unknown command {args.command}")
