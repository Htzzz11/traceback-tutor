"""Stream A: bug injection. One Mutation per bug type, registered in MUTATIONS.

AST-level mutations for runtime errors; text-level edits for syntax errors,
because a program with a syntax error can't be parsed into an AST.

Mutations see the whole file (program + test calls) and are told where the program
ends. Only wrong_arg_count may change a test call (as in its fixture); the others
only change the program.
"""

from __future__ import annotations

from . import missing_colon, misspelt_variable, off_by_one_index, str_plus_number, wrong_arg_count
from .base import Mutation, MutationResult

MUTATIONS: dict[str, Mutation] = {
    "misspelt_variable": misspelt_variable.apply,
    "str_plus_number": str_plus_number.apply,
    "off_by_one_index": off_by_one_index.apply,
    "wrong_arg_count": wrong_arg_count.apply,
    "missing_colon": missing_colon.apply,
}

__all__ = ["MUTATIONS", "Mutation", "MutationResult"]
