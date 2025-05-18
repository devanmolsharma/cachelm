import unittest
from cachelm.databases.database import Database
from cachelm.types.chat_history import Message


class TestDatabases(unittest.TestCase):
    def _test_helper(self, db: Database):
        """
        Test the database connection and basic operations.
        """
        db.reset()

        # Test writing to the database
        history = [
            Message(role="user", content="Hello, how are you?"),
            Message(role="assistant", content="I'm fine, thank you!"),
            Message(role="user", content="What can you do?"),
        ]
        response = Message(
            role="assistant",
            content="I can assist you with various tasks.",
        )
        db.write(history, response)

        # Test finding from the database
        result = db.find(history)
        assert result is not None, "Find result should not be None"
        assert (
            result.content == response.content
        ), "Find result should match written response"

    def test_chroma_database(self):
        """
        Test the Chroma database.
        """
        from cachelm.databases.chroma import ChromaDatabase
        from cachelm.vectorizers.fastembed import FastEmbedVectorizer

        vectorizer = FastEmbedVectorizer()
        db = ChromaDatabase(vectorizer)
        success = db.connect()
        assert success, "Failed to connect to Chroma database"
        self._test_helper(db)
        db.disconnect()

    def test_clickhouse_database(self):
        """
        Test the ClickHouse database.
        """
        from cachelm.databases.clickhouse import ClickHouse
        from cachelm.vectorizers.fastembed import FastEmbedVectorizer

        vectorizer = FastEmbedVectorizer()
        db = ClickHouse(
            host="localhost",
            port=18123,
            user="default",
            password="pass",
            vectorizer=vectorizer,
        )
        success = db.connect()
        assert success, "Failed to connect to ClickHouse database"
        self._test_helper(db)
        db.disconnect()

    def test_redisvl_database(self):
        """
        Test the RedisVL database.
        """
        from cachelm.databases.redisvl import RedisVLDatabase
        from cachelm.vectorizers.fastembed import FastEmbedVectorizer

        vectorizer = FastEmbedVectorizer()
        db = RedisVLDatabase(host="localhost", port=16379, vectorizer=vectorizer)
        success = db.connect()
        assert success, "Failed to connect to RedisVL database"
        self._test_helper(db)
        db.disconnect()
