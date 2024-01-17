import unittest
import torch
import pytest
from src.models.main import model, tokenizer


class TestModelTranslation(unittest.TestCase):
    @pytest.mark.filterwarnings("ignore:`as_target_tokenizer` is deprecated and will be removed in v5 of Transformers")
    def setUp(self):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @pytest.mark.filterwarnings("ignore:`as_target_tokenizer` is deprecated and will be removed in v5 of Transformers")
    def test_single_sentence(self):
        # Define a Danish sentence
        danish_sentence = "Hej, jeg er lige ankommet."

        # Tokenize the sentence
        inputs = self.tokenizer.encode_plus(danish_sentence, return_tensors="pt", padding=True, truncation=True)

        # Move the inputs to the GPU if available
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}

        # Generate the translation
        outputs = self.model.generate(inputs["input_ids"], max_length=100, num_beams=5, early_stopping=True)

        # No exceptions should be raised till here, so the test is considered passed
        self.assertIsNotNone(outputs)


if __name__ == "__main__":
    unittest.main()
