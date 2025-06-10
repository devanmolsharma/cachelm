from abc import ABC, abstractmethod
from loguru import logger


class Vectorizer(ABC):
    """
    Base class for all embedders.
    """

    def __init__(self, decay=0.4):
        """
        Initialize the vectorizer with a decay factor.
        Args:
            decay (float): The decay factor for embedding weights.
        """
        self.decay = decay
        self._embedding_dimension_cached = None

    def embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        Returns:
            int: The dimension of the embedding vectors.
        """
        if self._embedding_dimension_cached is None:
            temp_vector = self.embed("test")
            self._embedding_dimension_cached = len(temp_vector)
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

    def embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        Returns:
            int: The dimension of the embedding vectors.
        """
        if self._embedding_dimension_cached is None:
            temp_vector = self.embed("test")
            self._embedding_dimension_cached = len(temp_vector)
        return self._embedding_dimension_cached

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
        reversed_text = text[
            ::-1
        ]  # Reverse the order of messages to give more weight to recent messages
        logger.debug(
            f"Splitting chat history into {len(reversed_text)} messages for embedding."
        )
        embeddings = self.embed_many(reversed_text)
        if not embeddings:
            return []

        weighted_sum = [0.0] * len(embeddings[0])
        total_weight = 0.0

        for i, embedding in enumerate(embeddings):
            weight = self.decay**i
            total_weight += weight
            for j in range(len(weighted_sum)):
                weighted_sum[j] += embedding[j] * weight

        return [x / total_weight for x in weighted_sum] if total_weight > 0 else []

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
