class AggregateMethod:
    """
    Enum-like class to define aggregation methods.
    """

    EXPONENTIAL_DECAY = "exponential_decay"
    LINEAR_DECAY = "linear_decay"
    CONCATENATE = "concatenate"


class Aggregator:
    """Aggregator class to handle different aggregation methods for vectors.
    This class supports exponential decay, linear decay, and concatenation of vectors.
    It can aggregate a list of vectors based on the specified method and provides a method to get the effective embedding dimension based on the aggregation method.
    """

    def __init__(
        self, method: AggregateMethod, window_size: int = 4, decay: float = 0.4
    ):
        self.method = method
        if method not in [
            AggregateMethod.EXPONENTIAL_DECAY,
            AggregateMethod.LINEAR_DECAY,
            AggregateMethod.CONCATENATE,
        ]:
            raise ValueError(f"Invalid aggregation method: {method}")
        self.window_size = window_size
        self.decay = decay

    def get_effective_embedding_dimension(self, base_dim: int) -> int:
        if self.method == AggregateMethod.EXPONENTIAL_DECAY:
            return base_dim
        elif self.method == AggregateMethod.LINEAR_DECAY:
            return base_dim
        elif self.method == AggregateMethod.CONCATENATE:
            return base_dim * self.window_size

    def aggregate(self, vectors: list[list[float]]) -> list[float]:
        """
        Aggregate a list of vectors based on the specified method.
        Args:
            vectors (list[list[float]]): The list of vectors to aggregate.
        Returns:
            list[float]: The aggregated vector.
        """
        if not vectors:
            return []

        if self.method == AggregateMethod.EXPONENTIAL_DECAY:
            return self._exponential_decay(vectors)
        elif self.method == AggregateMethod.LINEAR_DECAY:
            return self._linear_decay(vectors)
        elif self.method == AggregateMethod.CONCATENATE:
            return self._concatenate(vectors)

    def _exponential_decay(self, vectors: list[list[float]]) -> list[float]:
        """
        Aggregate vectors using exponential decay.
        Args:
            vectors (list[list[float]]): The list of vectors to aggregate.
        Returns:
            list[float]: The aggregated vector.
        """
        weighted_sum = [0.0] * len(vectors[0])
        total_weight = 0.0

        for i, vector in enumerate(vectors):
            weight = self.decay**i
            total_weight += weight
            for j in range(len(weighted_sum)):
                weighted_sum[j] += vector[j] * weight

        return [x / total_weight for x in weighted_sum] if total_weight > 0 else []

    def _linear_decay(self, vectors: list[list[float]]) -> list[float]:
        """
        Aggregate vectors using linear decay.
        Args:
            vectors (list[list[float]]): The list of vectors to aggregate.
        Returns:
            list[float]: The aggregated vector.
        """
        weighted_sum = [0.0] * len(vectors[0])
        total_weight = 0.0

        for i, vector in enumerate(vectors):
            weight = self.window_size - i
            total_weight += weight
            for j in range(len(weighted_sum)):
                weighted_sum[j] += vector[j] * weight

        return [x / total_weight for x in weighted_sum] if total_weight > 0 else []

    def _concatenate(self, vectors: list[list[float]]) -> list[float]:
        """
        Concatenate vectors.
        Args:
            vectors (list[list[float]]): The list of vectors to concatenate.
        Returns:
            list[float]: The concatenated vector.
        """
        return [item for sublist in vectors for item in sublist]
