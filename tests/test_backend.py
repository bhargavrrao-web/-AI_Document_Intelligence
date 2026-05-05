import unittest
import sys
import os

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../backend')))

from quantum_logic import q_classifier

class TestQuantumLogic(unittest.TestCase):
    def test_quantum_analysis_output(self):
        # Create a dummy embedding (128 dimensions, though our circuit uses fewer)
        dummy_embedding = [0.1] * 128
        label, score = q_classifier.analyze_document(dummy_embedding)
        
        self.assertIsInstance(label, str)
        self.assertIsInstance(score, float)
        self.assertIn("Quantum", label)
        self.assertGreaterEqual(score, -1.0)
        self.assertLessEqual(score, 1.0)

if __name__ == '__main__':
    unittest.main()
