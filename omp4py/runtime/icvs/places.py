"""Parser for the OMP_PLACES environment variable.

This module implements a parser for the `OMP_PLACES` environment variable
as defined in the OpenMP specification. The syntax of `OMP_PLACES` is
complex and allows describing thread affinity using named places
(e.g., "cores", "threads") or explicit resource lists with intervals,
strides, and exclusions.

The parser converts the textual representation into an internal
`PlacePartition` structure used by the runtime to manage thread affinity.

Supported features include:
    - Named places (e.g., "cores", "threads", "sockets")
    - Explicit lists of hardware resources (e.g., "{0,1,2}")
    - Intervals with length and stride (e.g., "{0}:4:2")
    - Exclusions using the '!' operator
    - Combination of multiple place definitions

Invalid values are reported to stderr following OpenMP behavior.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Generator
from dataclasses import dataclass, field

# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.icvs import icvs
from omp4py.runtime.lowlevel.numeric import new_pyint_array

# END_CYTHON_IMPORTS

__all__ = ["parse"]

TOKEN_SPEC: list[tuple[str, str]] = [
    ("NUMBER", r"\d+"),
    ("WORD", r"[a-zA-Z_]+"),
    ("LBRACE", r"\{"),
    ("RBRACE", r"\}"),
    ("COLON", r":"),
    ("COMMA", r","),
    ("BANG", r"!"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("SKIP", r"\s+"),
]

TOKEN_RE: re.Pattern[str] = re.compile("|".join(f"(?P<{name}>{pattern})" for name, pattern in TOKEN_SPEC))


@dataclass
class Place:
    partition: list[int] = field(default_factory=list)
    exclusion: bool = False
    name: str = ""


def tokenize(text: str) -> Generator[tuple[str, str]]:
    for match in TOKEN_RE.finditer(text):
        if match.lastgroup and match.lastgroup != "SKIP":
            yield match.lastgroup, match.group()
    yield "END", ""


def check_exclusions(places: list[Place], merge: bool) -> list[Place]:
    new_places: list[Place] = []
    exclusions: set[int] = set()
    for place in places:
        if place.exclusion:
            exclusions.update(place.partition)
        elif not merge or len(new_places) == 0:
            new_places.append(place)
        else:
            new_places[0].partition += place.partition

    if exclusions:
        for place in new_places:
            place.partition = [value for value in place.partition if value not in exclusions]

    return new_places


class Parser:
    tokenizer: Generator[tuple[str, str]]
    token: tuple[str, str]

    def __init__(self, text: str) -> None:
        self.tokenizer = tokenize(text)
        self.token = next(self.tokenizer)

    def consume(self, expected: str) -> tuple[str, str]:
        if self.token[0] != expected:
            msg = f"Expected {expected}, got {self.token[1]}"
            raise ValueError(msg)
        old_token = self.token
        if self.token[0] != "END":
            self.token = next(self.tokenizer)
        return old_token

    # ⟨list⟩ |= ⟨p-list⟩ | ⟨aname⟩
    def parse_list(self) -> list[Place]:
        if self.token[0] == "WORD":
            return [self.parse_aname()]
        return check_exclusions(self.parse_plist(), merge=False)

    # ⟨aname⟩      |= ⟨word⟩(⟨num-places⟩) | ⟨word⟩
    # ⟨word⟩       |= sockets | cores | threads
    def parse_aname(self) -> Place:
        place = Place(name=self.token[1])
        if place.name not in ("threads", "cores", "sockets", "numa_domains", "ll_caches"):
            raise ValueError(place.name)
        self.consume("WORD")
        if self.token[0] == "LPAREN":
            self.consume("LPAREN")
            place.partition.append(max(int(self.consume("NUMBER")[1]), 1))
            self.consume("RPAREN")
        return place

    # ⟨p-list⟩ |= ⟨p-interval⟩ | ⟨p-list⟩,⟨p-interval⟩
    def parse_plist(self) -> list[Place]:
        places: list[Place] = [self.parse_pinterval()]
        while self.token[0] == "COMMA":
            self.consume("COMMA")
            places.append(self.parse_pinterval())
        return places

    # ⟨p-interval⟩ |= ⟨place⟩:⟨len⟩:⟨stride⟩ | ⟨place⟩:⟨len⟩ | ⟨place⟩ | !⟨place⟩
    def parse_pinterval(self) -> Place:
        exclusion = False
        if self.token[0] == "BANG":
            self.consume("BANG")
            exclusion = True

        place = self.parse_place()

        if self.token[0] == "COLON":
            self.consume("COLON")
            length = int(self.consume("NUMBER")[1])
            stride = 1
            if self.token[0] == "COLON":
                self.consume("COLON")
                stride = int(self.consume("NUMBER")[1])

            values: list[int] = list(place.partition)
            for i in range(length):
                values.extend([x + i * stride for x in place.partition])

            place.partition = values

        place.exclusion = exclusion
        return place

    # ⟨place⟩ |= {⟨res-list⟩}
    def parse_place(self) -> Place:
        self.consume("LBRACE")
        res = self.parse_reslist()
        self.consume("RBRACE")
        return check_exclusions(res, merge=True)[0]

    # ⟨res-list⟩ |= ⟨res-interval⟩ | ⟨res-list⟩,⟨res-interval⟩
    def parse_reslist(self) -> list[Place]:
        places: list[Place] = [self.parse_resinterval()]
        while self.token[0] == "COMMA":
            self.consume("COMMA")
            places.append(self.parse_resinterval())
        return places

    # ⟨res-interval⟩ |= ⟨res⟩:⟨num-places⟩:⟨stride⟩ | ⟨res⟩:⟨num-places⟩ | ⟨res⟩ | !⟨res⟩
    def parse_resinterval(self) -> Place:
        place = Place()

        if self.token[0] == "BANG":
            self.consume("BANG")
            place.exclusion = True
        else:
            place.exclusion = False

        res = int(self.consume("NUMBER")[1])
        if self.token[0] == "COLON":
            self.consume("COLON")
            num_places = int(self.consume("NUMBER")[1])
            stride = 1
            if self.token[0] == "COLON":
                self.consume("COLON")
                stride = int(self.consume("NUMBER")[1])
            place.partition.extend([res + stride * i for i in range(num_places)])
        else:
            place.partition.append(res)
        return place

    def parse(self) -> icvs.PlacePartition:
        places = self.parse_list()
        self.consume("END")

        result: icvs.PlacePartition = icvs.PlacePartition.__new__(icvs.PlacePartition)
        result.name = ""
        if len(places) == 1 and places[0].name:
            result.name = places[0].name
            result.partitions = new_pyint_array(0)
            if places[0].partition:
                result.values = new_pyint_array(1)
                result.values[0] = places[0].partition[0]
            else:
                result.values = new_pyint_array(0)
            return result

        values: list[int] = []
        partitions: list[int] = []
        for place in places:
            values.extend(place.partition)
            partitions.append(len(values))
        result.partitions = new_pyint_array(len(partitions))
        for i in range(len(partitions)):
            result.partitions[i] = partitions[i]
        result.values = new_pyint_array(len(values))
        for i in range(len(values)):
            result.values[i] = values[i]
        return result


def parse(var: str) -> icvs.PlacePartition | None:
    """Parse an OMP_PLACES environment variable.

    Reads the value of the given environment variable and parses it
    according to the OpenMP `OMP_PLACES` syntax.

    Args:
        var: Name of the environment variable (typically "OMP_PLACES").

    Returns:
        A `PlacePartition` object describing the parsed affinity
        configuration, or `None` if the variable is not set or invalid.

    Notes:
        If parsing fails, an error message is printed to stderr and
        `None` is returned.
    """
    try:
        if text := os.environ.get(var, None):
            return Parser(text).parse()
    except ValueError as ex:
        print(f"omp4py: Unknown value '{ex}' for environment variable {var}", file=sys.stderr)  # noqa: T201
    return None
