from cachelm.vectorizers.vectorizer import Vectorizer
import unittest


class TestVectorizer(unittest.TestCase):
    def _test_helper(self, vectorizer: Vectorizer):
        """
        Test the vectorizer.
        """
        text = "Hello, world!"
        embedding = vectorizer.embed(text)
        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) > 0, "Embedding should not be empty"
        assert isinstance(embedding[0], float), "Embedding should be a list of floats"

        multiple_texts = ["Hello, world!", "Goodbye, world!"]
        embeddings = vectorizer.embed_many(multiple_texts)
        assert isinstance(embeddings, list), "Embeddings should be a list"
        assert len(embeddings) == len(
            multiple_texts
        ), "Embeddings length should match input texts length"

        for embedding in embeddings:
            assert isinstance(embedding, list), "Each embedding should be a list"
            assert len(embedding) > 0, "Each embedding should not be empty"
            assert isinstance(
                embedding[0], float
            ), "Each embedding should be a list of floats"
            countSame = embeddings.count(embedding)
            assert countSame == 1, "Each embedding should be unique"
            assert len(embedding) == len(
                embeddings[0]
            ), "All embeddings should have the same length"

    def test_fastembed_vectorizer(self):
        """
        Test the FastEmbed vectorizer.
        """
        from cachelm.vectorizers.fastembed import FastEmbedVectorizer

        vectorizer = FastEmbedVectorizer()
        self._test_helper(vectorizer)

    def test_chroma_vectorizer(self):
        """
        Test the Chroma vectorizer.
        """
        from cachelm.vectorizers.chroma import ChromaVectorizer

        vectorizer = ChromaVectorizer()
        self._test_helper(vectorizer)

    def test_redisvl_vectorizer(self):
        """
        Test the RedisVL vectorizer.
        """
        from cachelm.vectorizers.redisvl import RedisvlVectorizer

        vectorizer = RedisvlVectorizer()
        self._test_helper(vectorizer)
