from loguru import logger

try:
    from redisvl.extensions.cache.llm import SemanticCache
    from redisvl.utils.vectorize import CustomTextVectorizer
    from cachelm.databases.database import Database
    from cachelm.vectorizers.vectorizer import Vectorizer
except ImportError:
    raise ImportError(
        "RedisVL library is not installed. Run `pip install redisvl` to install it."
    )


class RedisCache(Database):
    """
    Redis database for caching.
    """

    def __init__(
        self, host: str, port: int, vectorizer: Vectorizer, unique_id: str = "cachelm"
    ):
        super().__init__(vectorizer, unique_id)
        self.host = host
        self.port = port
        self.cache = None

    def connect(self) -> bool:
        """
        Connect to the Redis database.
        """
        try:
            self.cache = SemanticCache(
                redis_url=f"redis://{self.host}:{self.port}",
                vectorizer=CustomTextVectorizer(
                    embed=self.vectorizer.embed,
                    embed_many=self.vectorizer.embedMany,
                ),
                overwrite=True,
                name=self.unique_id,
            )
            return True
        except Exception as e:
            logger.error(f"Error connecting to Redis: {e}")
            return False

    def disconnect(self):
        """
        Disconnect from the Redis database.
        """
        if self.cache:
            self.cache.disconnect()

    def write(self, history: list[str], response: str):
        """
        Write data to the Redis database.
        """
        logger.info(f"Writing to Redis: {history} -> {response}")
        try:
            self.cache.store(
                prompt=" ".join(history),
                response=response,
            )
        except Exception as e:
            logger.error(f"Error writing to Redis: {e}")
            return

    def find(self, history: list[str], distance_threshold=0.2) -> str | None:
        """Find data in the database."""
        try:
            res = self.cache.check(
                prompt=" ".join(history),
                distance_threshold=distance_threshold,
            )
            if res is not None and len(res) > 0:
                logger.info(f"Found in Redis: {res[0].get('response','')[0:50]}...")
                return res[0].get("response", "")
            return
        except Exception as e:
            logger.error(f"Error finding from redis: {e}")
            return
