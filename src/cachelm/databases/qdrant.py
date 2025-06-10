from typing import Literal
from uuid import uuid4
from cachelm.types.chat_history import Message
from cachelm.databases.database import Database
from cachelm.vectorizers.vectorizer import Vectorizer
from loguru import logger

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams, PointStruct
except ImportError:
    raise ImportError(
        "Qdrant library is not installed. Run `pip install qdrant-client` to install it."
    )


class QdrantDatabase(Database):
    """
    Qdrant database for caching.
    """

    client = QdrantClient

    def __init__(
        self,
        vectorizer: Vectorizer,
        unique_id: str = "cachelm",
        distance: Distance = Distance.COSINE,
        location: str = ":memory:",
        collection_name: str = None,
        prefer_grpc: bool = True,
        timeout: int = None,
        api_key: str = None,
        host: str = None,
        port: int = None,
        grpc_port: int = None,
        https: bool = None,
        prefix: str = None,
        path: str = None,
        force_disable_check_same_thread: bool = False,
        grpc_options: dict = None,
        auth_token_provider: callable = None,
        cloud_inference: bool = False,
        local_inference_batch_size: int = None,
        check_compatibility: bool = True,
    ):
        """
        Initialize the Qdrant database.
        Args:
            vectorizer (Vectorizer): The vectorizer to use for embeddings.
            unique_id (str): Unique identifier for the collection.
            distance (Distance): Distance metric to use for similarity search.
            client_parameters (dict): Parameters for Qdrant client connection.
            collection_name (str): Name of the Qdrant collection (defaults to unique_id).
            prefer_grpc (bool): Whether to prefer gRPC protocol (default: True).
            timeout (int): Timeout for requests.
            api_key (str): API key for authentication.
            host (str): Host address of the Qdrant server.
            port (int): HTTP port of the Qdrant server.
            grpc_port (int): gRPC port of the Qdrant server.
            https (bool): Use HTTPS for connection.
            prefix (str): Prefix for REST API endpoints.
            path (str): Path for Unix socket connection.
            force_disable_check_same_thread (bool): Disable SQLite thread check.
            grpc_options (dict): Additional gRPC options.
            auth_token_provider (callable): Provider for authentication token.
            cloud_inference (bool): Enable cloud inference.
            local_inference_batch_size (int): Batch size for local inference.
            check_compatibility (bool): Check client-server compatibility.
        """
        super().__init__(vectorizer, unique_id)
        self.client = None
        self.collection_name = collection_name or unique_id
        self.distance = distance
        self.client_parameters = {}
        # Only add if not already present in client_parameters
        if location is not None and host is None:
            self.client_parameters["location"] = location
        if prefer_grpc is not None:
            self.client_parameters["prefer_grpc"] = prefer_grpc
        if timeout is not None:
            self.client_parameters["timeout"] = timeout
        if api_key is not None:
            self.client_parameters["api_key"] = api_key
        if host is not None:
            self.client_parameters["host"] = host
        if port is not None:
            self.client_parameters["port"] = port
        if grpc_port is not None:
            self.client_parameters["grpc_port"] = grpc_port
        if https is not None:
            self.client_parameters["https"] = https
        if prefix is not None:
            self.client_parameters["prefix"] = prefix
        if path is not None:
            self.client_parameters["path"] = path
        if force_disable_check_same_thread:
            self.client_parameters["force_disable_check_same_thread"] = (
                force_disable_check_same_thread
            )
        if grpc_options is not None:
            self.client_parameters["grpc_options"] = grpc_options
        if auth_token_provider is not None:
            self.client_parameters["auth_token_provider"] = auth_token_provider
        if cloud_inference:
            self.client_parameters["cloud_inference"] = cloud_inference
        if local_inference_batch_size is not None:
            self.client_parameters["local_inference_batch_size"] = (
                local_inference_batch_size
            )
        if check_compatibility is not None:
            self.client_parameters["check_compatibility"] = check_compatibility

    def connect(self) -> bool:
        try:
            self.client = QdrantClient(**self.client_parameters)
            # Create collection if not exists
            try:
                self.collection = self.client.get_collection(self.collection_name)
            except Exception:
                dim = self.vectorizer.embedding_dimension()
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=dim, distance=self.distance),
                )
                logger.info("Qdrant collection created.")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Qdrant: {e}")
            return False

    def disconnect(self):
        pass

    def reset(self):
        """
        Reset the database.
        """
        try:
            if self.client:
                self.client.delete_collection(self.collection_name)
                dim = self.vectorizer.embedding_dimension()
                self.client.recreate_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=dim, distance=self.distance),
                )
                logger.info("Qdrant database reset and reconnected.")
        except Exception as e:
            logger.error(f"Error resetting Qdrant: {e}")

    def write(self, history: list[Message], response: Message):
        logger.info(f"Writing to Qdrant: {history} -> {response}")
        try:
            history_strs = [msg.to_formatted_str() for msg in history]
            document = "\n".join(history_strs)
            embedding = self.vectorizer.embed_weighted_average(document)
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=str(uuid4()),
                        vector=embedding,
                        payload={
                            "document": document,
                            "response": response.to_json_str(),
                        },
                    )
                ],
            )
        except Exception as e:
            logger.error(f"Error writing to Qdrant: {e}")

    def find(self, history: list[Message], distance_threshold=0.2) -> Message | None:
        try:
            history_strs = [msg.to_formatted_str() for msg in history]
            document = "\n".join(history_strs)
            embedding = self.vectorizer.embed_weighted_average(document)
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=1,
                with_payload=True,
                score_threshold=(
                    1 - distance_threshold if self.distance == Distance.COSINE else None
                ),
            )
            if search_result:
                point = search_result[0]
                score = point.score
                logger.info(f"Qdrant search score: {score}")
                if self.distance == Distance.COSINE:
                    # Qdrant returns similarity, not distance, for cosine
                    if score < 1 - distance_threshold:
                        logger.info(
                            f"Score too low: {score} < {1 - distance_threshold}"
                        )
                        return
                else:
                    if score > distance_threshold:
                        logger.info(
                            f"Distance too high: {score} > {distance_threshold}"
                        )
                        return
                response_str = point.payload.get("response")
                if response_str is None:
                    logger.info("No response found")
                    return
                logger.info(f"Found in Qdrant: {response_str[:100]}...")
                return Message.from_json_str(response_str)
            logger.info("No match found in Qdrant.")
            return
        except Exception as e:
            logger.error(f"Error finding from Qdrant: {e}")
            return

    def size(self) -> int:
        """
        Get the size of the database.
        """
        try:
            info = self.client.get_collection(self.collection_name)
            return info.points_count
        except Exception as e:
            logger.error(f"Error getting size of Qdrant: {e}")
            return 0
