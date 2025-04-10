# tests/test_checker.py
import unittest
from checker import BERTPlagiarismChecker

class TestBERTPlagiarismChecker(unittest.TestCase):
    def setUp(self):
        self.checker = BERTPlagiarismChecker()

    def test_identical_texts(self):
        text1 = "The cat chased the mouse."
        text2 = "The cat chased the mouse."

        score, status = self.checker.check_similarity(text1, text2)
        self.assertGreaterEqual(score, 85)
        self.assertIn("Plagiarism", status)

    def test_different_texts(self):
        text1 = "The sun rises in the east."
        text2 = "Quantum computing is a new technology."
        score, status = self.checker.check_similarity(text1, text2)
        self.assertLess(score, 40)
        self.assertEqual(status, "Completely Different")
