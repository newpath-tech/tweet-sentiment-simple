import json
import unittest

from xquik_source import XquikSourceError, fetch_xquik_tweets, parse_tweet_texts


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class FakeOpener:
    def __init__(self, payload):
        self.payload = payload
        self.request = None
        self.timeout = None

    def __call__(self, request, timeout):
        self.request = request
        self.timeout = timeout
        return FakeResponse(self.payload)


class XquikSourceTest(unittest.TestCase):
    def test_parse_tweet_texts_supports_nested_payloads(self):
        payload = {
            "data": {
                "tweets": [
                    {"text": "Great launch today"},
                    {"legacy": {"full_text": "Needs more context"}},
                    {"text": "Ignored by limit"},
                ]
            }
        }

        self.assertEqual(
            parse_tweet_texts(payload, limit=2),
            ["Great launch today", "Needs more context"],
        )

    def test_fetch_tweets_sends_api_key_and_encoded_query(self):
        opener = FakeOpener({"tweets": [{"text": "Live X post"}]})

        tweets = fetch_xquik_tweets(
            "python lang:en",
            api_key="xq_test",
            opener=opener,
        )

        self.assertEqual(tweets, ["Live X post"])
        self.assertIsNotNone(opener.request)
        self.assertEqual(opener.request.get_header("X-api-key"), "xq_test")
        self.assertIn("q=python+lang%3Aen", opener.request.full_url)
        self.assertEqual(opener.timeout, 20)

    def test_fetch_tweets_requires_api_key(self):
        with self.assertRaisesRegex(XquikSourceError, "XQUIK_API_KEY"):
            fetch_xquik_tweets("python", api_key="", opener=FakeOpener({}))


if __name__ == "__main__":
    unittest.main()
