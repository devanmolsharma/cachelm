from abc import ABC, abstractmethod
from loguru import logger
from cachelm.utils.aggregator import AggregateMethod, Aggregator


class Vectorizer(ABC):
    """
    Base class for all embedders.
    """

    def __init__(
        self,
        decay=0.4,
        aggregate_method: AggregateMethod = AggregateMethod.CONCATENATE,
        window_size: int = 4,
    ):
        """
        Initialize the vectorizer with a decay factor.
        Args:
            decay (float): The decay factor for embedding weights.
            aggregate_method (AggregateMethod): The method to use for aggregating embeddings.
        """
        self.decay = decay
        self._embedding_dimension_cached = None
        self.aggregate_method = aggregate_method
        self.aggregator = Aggregator(
            aggregate_method, window_size=window_size, decay=decay
        )
        self.window_size = window_size

    def embedding_dimension(self, effective=True) -> int:
        """
        Get the dimension of the embedding vectors.
        This method caches the dimension after the first call to avoid repeated computation.
        Args:
            effective (bool): If True, returns the effective embedding dimension based on the aggregation method.
        Returns:
            int: The dimension of the embedding vectors.
        """
        if self._embedding_dimension_cached is None:
            temp_vector = self.embed("test")
            self._embedding_dimension_cached = len(temp_vector)

        if effective:
            return self.aggregator.get_effective_embedding_dimension(
                self._embedding_dimension_cached
            )
        else:
            return self._embedding_dimension_cached

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """
        Embed a single text string into a vector.
        Args:
            text (str): The text to embed.
        Returns:
            list[float]: The embedded vector.
        """
        raise NotImplementedError("embed method not implemented")

    @abstractmethod
    def embed_many(self, text: list[str]) -> list[list[float]]:
        """
        Embed multiple text strings into vectors.
        Args:
            text (list[str]): The list of texts to embed.
        Returns:
            list[list[float]]: The list of embedded vectors.
        """
        raise NotImplementedError("embed method not implemented")

    def embed_weighted_average(self, chatHistoryString: str) -> list[float]:
        """
        Embed a chat history string into a weighted average vector.
        This method takes a chat history string, splits it into individual messages,
        embeds each message, and computes a weighted average of the embeddings.
        The weighting is done using a decay factor, where the most recent message has the highest weight,
        and the weight decreases exponentially for older messages.
        This is useful for summarizing the chat history into a single vector representation,
        and makes it easier to handle long chat histories by focusing on the most recent messages.
        Args:
            text (list[str]): The list of texts to embed.
        Returns:
            list[float]: The weighted average embedded vector.
        """
        text = chatHistoryString.split("msg:")
        reversed_text = text[::-1][
            : self.window_size
        ]  # Reverse and limit to window size
        logger.debug(
            f"Splitting chat history into {len(reversed_text)} messages for embedding."
        )
        embeddings = self.embed_many(reversed_text)
        return self.aggregator.aggregate(embeddings)

    def embed_weighted_average_many(
        self, chatHistoryStrings: list[str]
    ) -> list[list[float]]:
        """
        Embed multiple chat history strings into weighted average vectors.
        This method takes a list of chat history strings, splits each into individual messages,
        embeds each message, and computes a weighted average of the embeddings for each chat history.
        Args:
            chatHistoryStrings (list[str]): The list of chat history strings to embed.
        Returns:
            list[list[float]]: The list of weighted average embedded vectors.
        """
        return [
            self.embed_weighted_average(chatHistoryString)
            for chatHistoryString in chatHistoryStrings
        ]
