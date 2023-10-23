"""Main method for interacting with denoiser through CLI.
"""

import sys
import spr_pick
import spr_pick.cli

from typing import List


def start_cli(args: List[str] = None):
    spr_pick.logging_helper.setup()
    if args is not None:
        sys.argv[1:] = args
    spr_pick.cli.start()


if __name__ == "__main__":
    start_cli()
