# SYSTEM IMPORTS
from collections import defaultdict
from enum import Enum
from typing import Dict, List, Tuple, Type
import csv
import os
from pprint import pprint


# PYTHON PROJECT IMPORTS


#################################### DO NOT CHANGE THIS CODE ####################################

# types defined in this module
ColorType: Type = Type["Color"]
SoftnessType: Type = Type["Softness"]
GoodToEatType: Type = Type["GoodToEat"]
AvacadoPredictorType: Type = Type["AvacadoPredictor"]


class Color(Enum):
    BLACK = 0
    BROWN = 1
    GREEN = 2

    @classmethod
    def value_of(cls: ColorType,
                 s: str
                 ) -> ColorType:
        for x in cls._member_names_:
            if s.upper() == x:
                return cls.__dict__[x]
        raise ValueError(f"ERROR: unknown string {s}")



class Softness(Enum):
    MUSHY = 0
    SOFT = 1
    TENDER = 2
    HARD = 3

    @classmethod
    def value_of(cls: ColorType,
                 s: str
                 ) -> ColorType:
        for x in cls._member_names_:
            if s.upper() == x:
                return cls.__dict__[x]
        raise ValueError(f"ERROR: unknown string {s}")


class GoodToEat(Enum):
    YES = 0
    NO = 1

    @classmethod
    def value_of(cls: ColorType,
                 s: str
                 ) -> ColorType:
        for x in cls._member_names_:
            if s.upper() == x:
                return cls.__dict__[x]
        raise ValueError(f"ERROR: unknown string {s}")


def load_data() -> List[Tuple[Color, Softness, GoodToEat]]:
    cd: str = os.path.dirname(__file__)
    data_dir: str = os.path.join(cd, "..", "data")
    if not os.path.exists(data_dir):
        raise Exception(f"ERROR: data_dir {data_dir} does not exist!")

    data_file: str = os.path.join(data_dir, "train_avacados.txt")
    if not os.path.exists(data_file):
        raise Exception(f"ERROR: file {data_file} does not exist!")

    data: List[Tuple[Color, Softness, GoodToEat]] = list()
    with open(data_file, "r") as f:
        reader = csv.reader(f, delimiter=",")

        for line in reader:
            if len(line) > 0:

                if len(line) != 3:
                    raise ValueError(f"ERROR: expected three values but got {line}")

                color, softness, good_to_eat = line
                data.append(tuple([Color.value_of(color.strip().rstrip()),
                                   Softness.value_of(softness.strip().rstrip()),
                                   GoodToEat.value_of(good_to_eat.strip().rstrip())]))

    return data

#################################### END OF "DO NOT CHANGE THIS CODE" ####################################




class AvacadoPredictor(object):
    def __init__(self: AvacadoPredictorType) -> None:
        self.color_given_good_to_eat_pmf: Dict[GoodToEat, Dict[Color, float]] = defaultdict(lambda: defaultdict(float))
        self.softness_given_good_to_eat_pmf: Dict[GoodToEat, Dict[Softness, float]] = defaultdict(lambda: defaultdict(float))
        self.good_to_eat_prior: Dict[GoodToEat, float] = defaultdict(float)


    def fit(self: AvacadoPredictorType,
            data: List[Tuple[Color, Softness, GoodToEat]]
            ) -> AvacadoPredictorType:

        # TODO: complete me!

        return self

    def predict_color_proba(self: AvacadoPredictorType,
                            X: List[Color]
                            ) -> List[List[Tuple[GoodToEat, float]]]:
        probs_per_example: List[List[Tuple[GoodToEat, float]]] = list()

        # TODO: complete me!

        return probs_per_example

    def predict_softness_proba(self: AvacadoPredictorType,
                               X: List[Softness]
                               ) -> List[List[Tuple[GoodToEat, float]]]:
        probs_per_example: List[List[Tuple[GoodToEat, float]]] = list()

        # TODO: complete me!

        return probs_per_example


    """
    # EXTRA CREDIT
    def predict_color(self: AvacadoPredictorType,
                      X: List[Color]
                      ) -> List[GoodToEat]:
        # TODO: complete me!
        return list()

    def predict_softness(self: AvacadoPredictorType,
                         X: List[Softness]
                         ) -> List[GoodToEat]:
        # TODO: complete me!
        return list()
    """




def accuracy(predictions: List[GoodToEat],
             actual: List[GoodToEat]
             ) -> float:
    if len(predictions) != len(actual):
        raise ValueError(f"ERROR: expected predictions and actual to be same length but got pred={len(predictions)}" +
            " and actual={len(actual)}")

    num_correct: float = 0
    for pred, act in zip(predictions, actual):
        num_correct += int(pred == act)

    return num_correct / len(predictions)


def main() -> None:
    data: List[Tuple[Color, Softness, GoodToEat]] = load_data()

    color_data: List[Color] = [color for color, _, _ in data]
    softness_data: List[Softness] = [softness for _, softness, _ in data]
    good_to_eat_data: List[GoodToEat] = [good_to_eat for _, _, good_to_eat in data]

    m: AvacadoPredictor = AvacadoPredictor().fit(data)

    print("good to eat prior")
    pprint(m.good_to_eat_prior)
    print()
    print()

    print("color given good to eat pmf")
    pprint(m.color_given_good_to_eat_pmf)
    print()
    print()

    print("softness given good to eat pmf")
    pprint(m.softness_given_good_to_eat_pmf)

    # if you do the extra credit be sure to uncomment these lines!
    # print("accuracy when predicting only on color: ", accuracy(m.predict_color(color_data), good_to_eat_data))

    # print("accuracy when predicting only on softness: ", accuracy(m.predict_softness(softness_data), good_to_eat_data))


if __name__ == "__main__":
    main()

