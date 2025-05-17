from loguru import logger

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
        """
        Initialize the ClickHouse database.
        Args:
            host (str): The host of the ClickHouse database.
            port (int): The port of the ClickHouse database.
            user (str): The user for the ClickHouse database.
            password (str): The password for the ClickHouse database.
            vectorizer (Vectorizer): The vectorizer to use.
            database (str): The name of the database.
            unique_id (str): The unique ID for the the chat.
        """
        super().__init__(vectorizer, unique_id)
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.client = None
        self.table = f"{self.database}.{self.unique_id}_cache"

    def connect(self) -> bool:
        """
        Connect to the ClickHouse database and create table if not exists.
        """
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

    def disconnect(self):
        """
        Disconnect from the ClickHouse database.
        """
        self.client = None

    def write(self, history: list[str], response: str):
        """
        Write data to the ClickHouse database.
        """
        prompt = " ".join(history)
        logger.info(f"Writing to ClickHouse: {prompt} -> {response}")
        try:
            embedding = self.vectorizer.embed(prompt)
            self.client.insert(
                self.table,
                [
                    [prompt, response, embedding],
                ],
                column_names=["prompt", "response", "embedding"],
            )
        except Exception as e:
            logger.error(f"Error writing to ClickHouse: {e}")

    def find(self, history: list[str], distance_threshold=0.2) -> str | None:
        """
        Find data in the ClickHouse database using cosine similarity.
        """
        try:
            prompt = " ".join(history)
            embedding = self.vectorizer.embed(prompt)
            # Use cosine similarity: 1 - cosine_distance < threshold
            query = f"""
                SELECT response, 
                    1 - (dotProduct(embedding, %(embedding)s) / (length(embedding) * length(%(embedding)s))) AS similarity
                FROM {self.table}
                ORDER BY similarity DESC
                LIMIT 1
            """
            result = self.client.query(query, parameters={"embedding": embedding})
            if result.result_rows:
                response, similarity = result.result_rows[0]
                if similarity >= (1 - distance_threshold):
                    logger.info(f"Found in ClickHouse: {response[0:50]}...")
                    return response
            return None
        except Exception as e:
            logger.error(f"Error finding from ClickHouse: {e}")
            return None
