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
    perturb_cli,
    slab_cli
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
    # slab
    parser_slab = subparsers.add_parser(
        "slab",
        help="Generate slabs from structures with optional random surface vacancies",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser_slab.add_argument(
        "ATOMS",
        type=str,
        nargs="+",
        help="the structure files to generate slabs, support multiple files.",
    )
    parser_slab.add_argument(
        "-m",
        "--miller-indices",
        type=str,
        nargs="+",
        required=True,
        help="Miller indices for slab generation."
             " Each set as a quoted string, e.g. --miller-indices '1 2 3' '1 0 0'",
    )
    parser_slab.add_argument(
        "--symprec",
        type=float,
        default=0.1,
        help="Symmetry precision for SpacegroupAnalyzer.",
    )
    parser_slab.add_argument(
        "--angle-tol",
        type=float,
        default=8,
        help="Angle tolerance for SpacegroupAnalyzer.",
    )
    parser_slab.add_argument(
        "--min-slab-ab",
        type=float,
        default=12.0,
        help="Minimum slab size in a and b directions after supercell construction.",
    )
    parser_slab.add_argument(
        "--min-slab",
        type=float,
        default=12.0,
        help="Minimum slab thickness in c direction.",
    )
    parser_slab.add_argument(
        "--min-vac",
        type=float,
        default=20.0,
        help="Minimum vacuum thickness in c direction.",
    )
    parser_slab.add_argument(
        "--max-normal-search",
        type=int,
        default=20,
        help="Maximum integer supercell factor to search for a normal c direction.",
    )
    parser_slab.add_argument(
        "--symmetrize-slab",
        action="store_true",
        help="Whether to symmetrize the slab.",
    )
    parser_slab.add_argument(
        "--no-tasker2-modify-polar",
        action="store_false",
        dest="tasker2_modify_polar",
        help="Whether not to apply Tasker 2 modification to polar slabs.",
    )
    parser_slab.add_argument(
        "--no-drop-polar",
        action="store_false",
        dest="drop_polar",
        help="Whether not to drop polar slabs after Tasker modification.",
    )
    parser_slab.add_argument(
        "--remove-atom-types",
        type=str,
        nargs="+",
        default=None,
        help="List of atom types (as strings) that can be removed. If None, all atom types can be removed.",
    )
    parser_slab.add_argument(
        "--min-vacancy-ratio",
        type=float,
        default=0.0,
        help="Minimum vacancy ratio on surface sites.",
    )
    parser_slab.add_argument(
        "--max-vacancy-ratio",
        type=float,
        default=0.3,
        help="Maximum vacancy ratio on surface sites.",
    )
    parser_slab.add_argument(
        "--num-vacancy-ratios",
        type=int,
        default=1,
        help="Number of vacancy ratios to sample between min and max.",
    )
    parser_slab.add_argument(
        "--n-sample-per-ratio",
        type=int,
        default=5,
        help="Number of random samples to generate per vacancy ratio.",
    )
    parser_slab.add_argument(
        "--surface-mapping-fractol",
        type=float,
        default=1e-5,
        help="Fractional coordinate tolerance for symmetry mapping.",
    )
    parser_slab.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser_slab.add_argument(
        "--detect-isolated-atom-range",
        type=float,
        default=3.0,
        help="Distance range to detect isolated atoms after removal.",
    )
    parser_slab.add_argument(
        "--no-remove-isolated-atom",
        action="store_false",
        dest="remove_isolated_atom",
        help="Whether not to remove isolated atoms after site removal.",
    )
    parser_slab.add_argument(
        "--max-return-slabs",
        type=int,
        default=500,
        help="Maximum number of slabs to return. If more structures are generated, random sampling is applied.",
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
    # Process miller_indices for slab command
    if getattr(args, 'command', None) == 'slab':
        try:
            processed = []
            for s in args.miller_indices:
                if isinstance(s, str):
                    processed.append(tuple(map(int, s.split())))
                elif isinstance(s, (tuple, list)) and len(s) == 3:
                    processed.append(tuple(map(int, s)))
                else:
                    raise ValueError(f"Invalid miller index: {s}")
            args.miller_indices = processed
        except Exception as e:
            raise ValueError(
                f"Failed to parse --miller-indices: {args.miller_indices}."
                f" Each must be a string of three integers, e.g. '1 2 3'."
            ) from e

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
    elif args.command == "slab":
        slab_cli(
            atoms_path_ls=args.ATOMS,
            miller_indices=args.miller_indices,
            symprec=args.symprec,
            angle_tol=args.angle_tol,
            min_slab_ab=args.min_slab_ab,
            min_slab=args.min_slab,
            min_vac=args.min_vac,
            max_normal_search=args.max_normal_search,
            symmetrize_slab=args.symmetrize_slab,
            tasker2_modify_polar=args.tasker2_modify_polar,
            drop_polar=args.drop_polar,
            remove_atom_types=args.remove_atom_types,
            min_vacancy_ratio=args.min_vacancy_ratio,
            max_vacancy_ratio=args.max_vacancy_ratio,
            num_vacancy_ratios=args.num_vacancy_ratios,
            n_sample_per_ratio=args.n_sample_per_ratio,
            surface_mapping_fractol=args.surface_mapping_fractol,
            seed=args.seed,
            detect_isolated_atom_range=args.detect_isolated_atom_range,
            remove_isolated_atom=args.remove_isolated_atom,
            max_return_slabs=args.max_return_slabs,
        )
    elif args.command is None:
        pass
    else:
        raise RuntimeError(f"unknown command {args.command}")
