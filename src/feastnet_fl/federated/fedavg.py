from collections import OrderedDict
from typing import List, Tuple

import torch


StateDictWithSize = Tuple[OrderedDict, int]


def weighted_fedavg(state_dicts: List[StateDictWithSize]) -> OrderedDict:
    if not state_dicts:
        raise ValueError("state_dicts must not be empty")

    total_examples = sum(num_examples for _, num_examples in state_dicts)
    aggregated = OrderedDict()
    largest_client_state = max(state_dicts, key=lambda item: item[1])[0]

    for key in state_dicts[0][0].keys():
        first_value = state_dicts[0][0][key]
        if torch.is_floating_point(first_value):
            weighted_sum = sum(sd[key] * (count / total_examples) for sd, count in state_dicts)
            aggregated[key] = weighted_sum.clone()
        else:
            aggregated[key] = largest_client_state[key].clone()

    return aggregated
