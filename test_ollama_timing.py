import unittest

from ollama_client import (
    get_ollama_timing_summary,
    get_ollama_timings,
    reset_ollama_timings,
)


class OllamaTimingTest(unittest.TestCase):
    def test_empty_profile_has_zero_totals(self):
        reset_ollama_timings()

        self.assertEqual(get_ollama_timings(), [])
        self.assertEqual(
            get_ollama_timing_summary(1.25),
            {
                "total_llm_calls": 0,
                "stage_timings": [],
                "total_llm_elapsed_seconds": 0.0,
                "total_request_elapsed_seconds": 1.25,
            },
        )


if __name__ == "__main__":
    unittest.main()
