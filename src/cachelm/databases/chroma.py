from uuid import uuid4
from cachelm.databases.database import Database
from cachelm.vectorizers.vectorizer import Vectorizer
from loguru import logger

try:
    import chromadb
except ImportError:
    raise ImportError(
        "ChromaDB library is not installed. Run `pip install chromadb` to install it."
    )


class ChromaDatabase(Database):
    """
    Chroma database for caching.
    """

    def __init__(
        self, vectorizer: Vectorizer, persistant=True, unique_id: str = "cachelm"
    ):
        """
        Initialize the Chroma database.
        Args:
            vectorizer (Vectorizer): The vectorizer to use.
            persistant (bool): Whether to use a persistant client.
            unique_id (str): The unique ID for the database.
        """
        super().__init__(vectorizer, unique_id)
        self.client = None
        self.collection = None
        self.persistant = persistant

    def __get_adapted_embedding_function(self, vectorizer: Vectorizer):
        """
        Get the adapted embedding function for Chroma.
        """

        class AdaptedEmbeddingFunction(chromadb.EmbeddingFunction):
            def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
                # embed the documents somehow
                return vectorizer.embedMany(input)

        return AdaptedEmbeddingFunction()

    def connect(self) -> bool:
        """
        Connect to the Chroma database.
        """
        try:
            self.client = (
                chromadb.Client()
                if not self.persistant
                else chromadb.PersistentClient()
            )
            self.collection = self.client.get_or_create_collection(
                self.unique_id,
                embedding_function=self.__get_adapted_embedding_function(
                    self.vectorizer
                ),
            )
            return True
        except Exception as e:
            logger.error(f"Error connecting to Chroma: {e}")
            return False

    def disconnect(self):
        """
        Disconnect from the Chroma database.
        """
        pass

    def write(self, history: list[str], response: str):
        """
        Write data to the Chroma database.
        """
        logger.info(f"Writing to Chroma: {history} -> {response}")
        try:
            self.collection.add(
                ids=str(uuid4()),
                documents=" ".join(history),
                metadatas={"response": response},
            )
        except Exception as e:
            logger.error(f"Error writing to Chroma: {e}")
            return

    def find(self, history: list[str], distance_threshold=0.2) -> str | None:
        """Find data in the database."""
        try:
            res = self.collection.query(query_texts=" ".join(history), n_results=1)
            # if res is not None and res.
            if res is not None and len(res.get("ids", [[]])[0]) > 0:
                distance = res.get("distances", [[1.0]])[0][0]
                if distance > distance_threshold:
                    logger.info(f"Distance too high: {distance} > {distance_threshold}")
                    return
                text = res.get("metadatas", [[{}]])[0][0].get("response", None)
                if text is None:
                    logger.info("No text found")
                    return
                logger.info(f"Found in Chroma: {text[0:50]}...")
                return text
            logger.info(f"Found in Chroma: {res}...")
            return
        except Exception as e:
            logger.error(f"Error finding from Chroma: {e}")
            return
