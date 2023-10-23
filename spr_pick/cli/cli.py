import argparse

from spr_pick.version import __version__

from spr_pick.cli.cmds.train import TrainCommand
from spr_pick.cli.cmds.eval import EvaluateCommand


def start():
    parser = argparse.ArgumentParser(
        prog="joint",
        description=(
            "Command line interface for the denoising training and evaluation system. "
            + "Supported algorithms include Self Supervised Denoising (joint)"
        ),
    )
    parser.add_argument(
        "--version", action="version", version="%(prog)s v" + __version__
    )

    cmd_parsers = parser.add_subparsers(dest="command", required=True)
    # Populate available commands
    cmd_list = [
        TrainCommand(),
        EvaluateCommand(),
    ]

    # Add commands to parser
    cmds = {}
    for cmd in cmd_list:
        cmd.configure(cmd_parsers)
        cmds[cmd.cmd()] = cmd
    # Process arguments
    args = parser.parse_args()
    arg_dict = vars(args)
    arg_dict["PARSER"] = parser

    # Call handle on function
    cmds[args.command].execute(arg_dict)


if __name__ == "__main__":
    start()
