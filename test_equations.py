import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from yelp_sentiment_recommender import calculate_ubeb_cbeb


class EmotionalBiasEquationTests(unittest.TestCase):
    def test_ubeb_cbeb_example_from_paper(self):
        # Table example values for U1/R1 and other user/restaurant offsets.
        rows = [
            ("U1", "R1", 3.0, -0.5),
            ("U1", "R2", 4.0, 0.25),
            ("U1", "R3", 3.0, -0.75),
            ("U1", "R4", 4.0, 0.0),
            ("U1", "R5", 4.0, 0.5),
            ("U1", "R6", 3.0, 0.2),
            ("U2", "R1", 3.0, 0.6),
            ("U3", "R1", 4.0, 0.5),
            ("U4", "R1", 4.0, 0.8),
            ("U5", "R1", 4.0, 0.6),
            ("U6", "R1", 2.0, -0.2),
        ]
        df = pd.DataFrame(
            rows, columns=["user_id", "business_id", "stars", "emotional_offset"]
        )
        df["review_id"] = [str(i) for i in range(len(df))]
        out = calculate_ubeb_cbeb(df)
        u1r1 = out[(out.user_id == "U1") & (out.business_id == "R1")].iloc[0]
        self.assertAlmostEqual(u1r1.ubeb_bias, -0.45, places=6)
        self.assertAlmostEqual(u1r1.cbeb_bias, -0.75, places=6)
        self.assertAlmostEqual(u1r1.rating_ubeb, 3.45, places=6)
        self.assertAlmostEqual(u1r1.rating_cbeb, 3.75, places=6)


if __name__ == "__main__":
    unittest.main()
