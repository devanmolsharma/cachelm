from loguru import logger

from cachelm.types.chat_history import Message  # Correct import

try:
    import clickhouse_connect
    from cachelm.databases.database import Database
    from cachelm.vectorizers.vectorizer import Vectorizer
except ImportError:
    raise ImportError(
        "clickhouse-connect library is not installed. Run `pip install clickhouse-connect` to install it."
    )


class ClickHouse(Database):
    """
    ClickHouse database for caching.
    """

    def __init__(
        self,
        host: str,
        port: int,
        user: str,
        password: str,
        vectorizer: Vectorizer,
        database: str = "cachelm",
        unique_id: str = "cachelm",
    ):
        super().__init__(vectorizer, unique_id)
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.client = None
        self.table = f"{self.database}.{self.unique_id}_cache"

    def connect(self) -> bool:
        try:
            self.client = clickhouse_connect.get_client(
                host=self.host,
                port=self.port,
                username=self.user,
                password=self.password,
                database="default",
            )
            self.client.command(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            self.client.command(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id UUID DEFAULT generateUUIDv4(),
                    prompt String,
                    response String,
                    embedding Array(Float32)
                ) ENGINE = MergeTree()
                ORDER BY id
                """
            )
            return True
        except Exception as e:
            logger.error(f"Error connecting to ClickHouse: {e}")
            return False

    def reset(self):
        """
        Reset the ClickHouse database.
        """
        try:
            self.client.command(f"DROP TABLE IF EXISTS {self.table}")
            logger.info("ClickHouse database reset.")
            self.client.command(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    id UUID DEFAULT generateUUIDv4(),
                    prompt String,
                    response String,
                    embedding Array(Float32)
                ) ENGINE = MergeTree()
                ORDER BY id
                """
            )
        except Exception as e:
            logger.error(f"Error resetting ClickHouse: {e}")

    def disconnect(self):
        """
        Disconnect from the ClickHouse database.
        """
        self.client = None

    def write(self, history: list[Message], response: Message):
        """
        Write data to the ClickHouse database.
        """
        # Serialize history as a JSON string of message JSONs
        prompt = "\n".join([msg.to_formatted_str() for msg in history])
        response_str = response.to_json_str()
        logger.info(f"Writing to ClickHouse: {prompt} -> {response_str}")
        try:
            # For embedding, you may want to use only the text content
            prompt_text = " ".join([msg.content for msg in history])
            embedding = self.vectorizer.embed(prompt_text)
            self.client.insert(
                self.table,
                [
                    [prompt, response_str, embedding],
                ],
                column_names=["prompt", "response", "embedding"],
            )
        except Exception as e:
            logger.error(f"Error writing to ClickHouse: {e}")

    def find(self, history: list[Message], distance_threshold=0.2) -> Message | None:
        """
        Find data in the ClickHouse database using cosine similarity.
        """
        try:
            prompt_text = "\n".join([msg.to_formatted_str() for msg in history])
            embedding = self.vectorizer.embed(prompt_text)
            logger.debug(f"Finding in ClickHouse: {prompt_text}")
            query = f"""
                SELECT response, 
                    1 - (dotProduct(embedding, %(embedding)s) / (length(embedding) * length(%(embedding)s))) AS similarity
                FROM {self.table}
                ORDER BY similarity DESC
                LIMIT 1
            """
            result = self.client.query(query, parameters={"embedding": embedding})
            if result.result_rows and len(result.result_rows) > 0:
                response_str, similarity = result.result_rows[0]
                if similarity >= (1 - distance_threshold):
                    logger.info(f"Found in ClickHouse: {response_str[0:50]}...")
                    return Message.from_json_str(response_str)
            return None
        except Exception as e:
            logger.error(f"Error finding from ClickHouse: {e}")
            return None
