from typing import List, Type

from src.use_cases.use_case import UseCase
from src.metrics import AggregatedL2
from src.metrics import Metric



class WinProbEstimation(UseCase):
    """
    Toy use case for testing purposes.

    """
    @property
    def name(self) -> str:
        return "win_prob_estimation_use_case"

    @property
    def target_columns(self) -> List[str]:
        return []

    @property
    def metrics(self) -> List[Type[Metric]]:
        return ['contingent_revenue','bid_price','nb_bids','nb_wins']