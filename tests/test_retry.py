"""
Tests for retry wrapper + idempotency key handling.

Mocks urllib.request to capture outgoing requests and simulate failures.
"""

import json
import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO
import urllib.error

from enyal_sdk.client import (
    _api_call, archive, prove, disclose, timestamp, create_agreement,
    compliance_attest, send_message, request_client_disclosure,
    request_share_proof, synthesise_knowledge,
    _IDEMPOTENCY_MAP,
)

API_KEY = "test-key-xxx"
BASE_URL = "http://test.local"


def _make_response(body_dict, status=200):
    """Create a mock urllib response."""
    resp = MagicMock()
    resp.status = status
    resp.read.return_value = json.dumps(body_dict).encode()
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _make_http_error(code, body_dict=None):
    """Create a urllib.error.HTTPError."""
    body = json.dumps(body_dict or {"detail": f"Error {code}"}).encode()
    fp = BytesIO(body)
    err = urllib.error.HTTPError(
        url="http://test.local/api/v1/test",
        code=code,
        msg=f"HTTP {code}",
        hdrs=MagicMock(),
        fp=fp,
    )
    return err


class TestRetrySuccess(unittest.TestCase):
    """1a. Successful first call returns immediately, no retries."""

    @patch("enyal_sdk.client.urllib.request.urlopen")
    def test_no_retry_on_success(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({"ok": True})
        result = _api_call(API_KEY, "GET", "/api/v1/search", base_url=BASE_URL)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_urlopen.call_count, 1)


class TestRetryOnNetworkError(unittest.TestCase):
    """1b. Network error → retry → success on 2nd attempt."""

    @patch("enyal_sdk.client.time.sleep")
    @patch("enyal_sdk.client.urllib.request.urlopen")
    def test_network_error_then_success(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = [
            urllib.error.URLError("Connection refused"),
            _make_response({"ok": True}),
        ]
        result = _api_call(API_KEY, "GET", "/api/v1/search", base_url=BASE_URL)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_urlopen.call_count, 2)
        self.assertEqual(mock_sleep.call_count, 1)


class TestRetryMultipleFailures(unittest.TestCase):
    """1c. 500 → retry → 500 → retry → success on 3rd."""

    @patch("enyal_sdk.client.time.sleep")
    @patch("enyal_sdk.client.urllib.request.urlopen")
    def test_two_500s_then_success(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = [
            _make_http_error(500),
            _make_http_error(502),
            _make_response({"ok": True}),
        ]
        result = _api_call(API_KEY, "GET", "/api/v1/search", base_url=BASE_URL)
        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_urlopen.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)


class TestRetryExhaustion(unittest.TestCase):
    """1d. 500 x max_retries → raises final error."""

    @patch("enyal_sdk.client.time.sleep")
    @patch("enyal_sdk.client.urllib.request.urlopen")
    def test_all_retries_exhausted(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = [
            _make_http_error(500) for _ in range(4)
        ]
        with self.assertRaises(RuntimeError) as ctx:
            _api_call(API_KEY, "GET", "/api/v1/search", base_url=BASE_URL)
        self.assertIn("500", str(ctx.exception))
        self.assertEqual(mock_urlopen.call_count, 4)  # 1 + 3 retries


class TestRetryAfterHeader(unittest.TestCase):
    """1e. 429 with Retry-After: retries respecting the header."""

    @patch("enyal_sdk.client.time.sleep")
    @patch("enyal_sdk.client.urllib.request.urlopen")
    def test_429_retry_after(self, mock_urlopen, mock_sleep):
        err_429 = _make_http_error(429, {"detail": "Rate limited"})
        err_429.headers = MagicMock()
        err_429.headers.get.return_value = "2.0"
        mock_urlopen.side_effect = [
            err_429,
            _make_response({"ok": True}),
        ]
        result = _api_call(API_KEY, "GET", "/api/v1/search", base_url=BASE_URL)
        self.assertEqual(result, {"ok": True})
        # Verify sleep was called with a delay >= 2.0 (Retry-After) + jitter
        delay_arg = mock_sleep.call_args[0][0]
        self.assertGreaterEqual(delay_arg, 2.0)


class TestNoRetryOn4xx(unittest.TestCase):
    """1f. 4xx (401, 403, 404, 422) → raises immediately, no retry."""

    @patch("enyal_sdk.client.time.sleep")
    @patch("enyal_sdk.client.urllib.request.urlopen")
    def test_401_no_retry(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = _make_http_error(401)
        with self.assertRaises(RuntimeError):
            _api_call(API_KEY, "GET", "/api/v1/search", base_url=BASE_URL)
        self.assertEqual(mock_urlopen.call_count, 1)
        mock_sleep.assert_not_called()

    @patch("enyal_sdk.client.time.sleep")
    @patch("enyal_sdk.client.urllib.request.urlopen")
    def test_422_no_retry(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = _make_http_error(422)
        with self.assertRaises(RuntimeError):
            _api_call(API_KEY, "GET", "/api/v1/search", base_url=BASE_URL)
        self.assertEqual(mock_urlopen.call_count, 1)


class TestUserProvidedKeyStable(unittest.TestCase):
    """1g. User-provided idempotency_key is used verbatim on all retries."""

    @patch("enyal_sdk.client.time.sleep")
    @patch("enyal_sdk.client.urllib.request.urlopen")
    def test_user_key_stable_across_retries(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = [
            _make_http_error(500),
            _make_http_error(502),
            _make_response({"ok": True}),
        ]
        _api_call(API_KEY, "POST", "/api/v1/prove",
                  body={"resource_type": "test"}, base_url=BASE_URL,
                  idempotency_key="user-key-abc-123-def-456-ghi-789-jkl")
        # Inspect all 3 outgoing requests
        self.assertEqual(mock_urlopen.call_count, 3)
        sent_keys = set()
        for call in mock_urlopen.call_args_list:
            req = call[0][0]
            body = json.loads(req.data)
            sent_keys.add(body.get("idempotency_key"))
        # All attempts must use the exact same user-provided key
        self.assertEqual(len(sent_keys), 1)
        self.assertEqual(sent_keys.pop(), "user-key-abc-123-def-456-ghi-789-jkl")


class TestAutoGeneratedKeyStable(unittest.TestCase):
    """1h. SDK-generated key is stable across retries within one call."""

    @patch("enyal_sdk.client.time.sleep")
    @patch("enyal_sdk.client.urllib.request.urlopen")
    def test_auto_key_stable_across_retries(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = [
            _make_http_error(500),
            _make_http_error(502),
            _make_response({"ok": True}),
        ]
        _api_call(API_KEY, "POST", "/api/v1/prove",
                  body={"resource_type": "test"}, base_url=BASE_URL)
        self.assertEqual(mock_urlopen.call_count, 3)
        sent_keys = set()
        for call in mock_urlopen.call_args_list:
            req = call[0][0]
            body = json.loads(req.data)
            sent_keys.add(body.get("idempotency_key"))
        # All attempts must use the same auto-generated key
        self.assertEqual(len(sent_keys), 1)
        key = sent_keys.pop()
        self.assertIsNotNone(key)
        # Must be a valid UUID4 format
        self.assertEqual(len(key), 36)


class TestDifferentCallsDifferentKeys(unittest.TestCase):
    """1i. Different calls generate different keys."""

    @patch("enyal_sdk.client.urllib.request.urlopen")
    def test_different_keys_per_call(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({"ok": True})
        _api_call(API_KEY, "POST", "/api/v1/prove",
                  body={"resource_type": "t1"}, base_url=BASE_URL)
        _api_call(API_KEY, "POST", "/api/v1/prove",
                  body={"resource_type": "t2"}, base_url=BASE_URL)
        self.assertEqual(mock_urlopen.call_count, 2)
        keys = []
        for call in mock_urlopen.call_args_list:
            req = call[0][0]
            body = json.loads(req.data)
            keys.append(body.get("idempotency_key"))
        self.assertNotEqual(keys[0], keys[1])


class TestRetryDisabled(unittest.TestCase):
    """retry=False disables all retries."""

    @patch("enyal_sdk.client.time.sleep")
    @patch("enyal_sdk.client.urllib.request.urlopen")
    def test_no_retry_when_disabled(self, mock_urlopen, mock_sleep):
        mock_urlopen.side_effect = _make_http_error(500)
        with self.assertRaises(RuntimeError):
            _api_call(API_KEY, "GET", "/api/v1/search", base_url=BASE_URL, retry=False)
        self.assertEqual(mock_urlopen.call_count, 1)
        mock_sleep.assert_not_called()


class TestIdempotencyKeyTranslation(unittest.TestCase):
    """Verify idempotency_key kwarg translates to correct server-side field name."""

    @patch("enyal_sdk.client.urllib.request.urlopen")
    def _call_and_capture(self, func, kwargs, mock_urlopen):
        """Helper: call func, return captured request body."""
        mock_urlopen.return_value = _make_response({"ok": True, "chunk_id": "x"})
        func(**kwargs)
        req = mock_urlopen.call_args[0][0]
        return json.loads(req.data) if req.data else {}

    def test_archive_sends_client_chunk_id(self):
        body = self._call_and_capture(archive, {
            "api_key": API_KEY, "agent_id": "a", "chunk_type": "t",
            "chunk_key": "k", "data": {"x": 1}, "base_url": BASE_URL,
            "idempotency_key": "user-key-123456789012345678901234567",
        })
        self.assertEqual(body["client_chunk_id"], "user-key-123456789012345678901234567")
        self.assertNotIn("idempotency_key", body)

    def test_timestamp_sends_client_chunk_id(self):
        body = self._call_and_capture(timestamp, {
            "api_key": API_KEY, "payload": "data", "base_url": BASE_URL,
            "idempotency_key": "user-key-123456789012345678901234567",
        })
        self.assertEqual(body["client_chunk_id"], "user-key-123456789012345678901234567")
        self.assertNotIn("idempotency_key", body)

    def test_create_agreement_sends_client_chunk_id(self):
        body = self._call_and_capture(create_agreement, {
            "api_key": API_KEY, "terms": "t", "parties": ["a"],
            "base_url": BASE_URL,
            "idempotency_key": "user-key-123456789012345678901234567",
        })
        self.assertEqual(body["client_chunk_id"], "user-key-123456789012345678901234567")
        self.assertNotIn("idempotency_key", body)

    def test_compliance_attest_sends_client_attestation_id(self):
        body = self._call_and_capture(compliance_attest, {
            "api_key": API_KEY, "period_start": "2026-01-01",
            "period_end": "2026-03-31", "systems": ["s1"],
            "base_url": BASE_URL,
            "idempotency_key": "user-key-123456789012345678901234567",
        })
        self.assertEqual(body["client_attestation_id"], "user-key-123456789012345678901234567")
        self.assertNotIn("idempotency_key", body)

    def test_prove_sends_idempotency_key(self):
        body = self._call_and_capture(prove, {
            "api_key": API_KEY, "resource_type": "test",
            "base_url": BASE_URL,
            "idempotency_key": "user-key-123456789012345678901234567",
        })
        self.assertEqual(body["idempotency_key"], "user-key-123456789012345678901234567")

    def test_disclose_sends_idempotency_key(self):
        body = self._call_and_capture(disclose, {
            "api_key": API_KEY, "chunk_ids": ["c1"],
            "recipient_pubkey_hex": "abc", "purpose": "test",
            "base_url": BASE_URL,
            "idempotency_key": "user-key-123456789012345678901234567",
        })
        self.assertEqual(body["idempotency_key"], "user-key-123456789012345678901234567")

    def test_request_client_disclosure_sends_idempotency_key(self):
        body = self._call_and_capture(request_client_disclosure, {
            "api_key": API_KEY, "base_url": BASE_URL,
            "chunk_ids": ["c1"], "purpose": "test",
            "idempotency_key": "user-key-123456789012345678901234567",
        })
        self.assertEqual(body["idempotency_key"], "user-key-123456789012345678901234567")

    def test_request_share_proof_sends_idempotency_key(self):
        body = self._call_and_capture(request_share_proof, {
            "api_key": API_KEY, "base_url": BASE_URL,
            "customer_share_hex": "deadbeef",
            "idempotency_key": "user-key-123456789012345678901234567",
        })
        self.assertEqual(body["idempotency_key"], "user-key-123456789012345678901234567")

    def test_send_message_sends_idempotency_key(self):
        body = self._call_and_capture(send_message, {
            "api_key": API_KEY, "sender_agent_id": "s",
            "thread_id": "t", "recipient_agent_id": "r",
            "message_type": "text", "payload": {"msg": "hi"},
            "base_url": BASE_URL,
            "idempotency_key": "user-key-123456789012345678901234567",
        })
        self.assertEqual(body["idempotency_key"], "user-key-123456789012345678901234567")


class TestSynthesiseKnowledgeBlocked(unittest.TestCase):
    """synthesise_knowledge raises immediately without making HTTP call."""

    @patch("enyal_sdk.client.urllib.request.urlopen")
    def test_raises_without_http_call(self, mock_urlopen):
        with self.assertRaises(RuntimeError) as ctx:
            synthesise_knowledge(API_KEY, "query", ["node1"])
        self.assertIn("session auth", str(ctx.exception))
        mock_urlopen.assert_not_called()


class TestGetEndpointsNoIdempotencyKey(unittest.TestCase):
    """GET endpoints don't get idempotency keys injected."""

    @patch("enyal_sdk.client.urllib.request.urlopen")
    def test_get_no_key(self, mock_urlopen):
        mock_urlopen.return_value = _make_response({"results": []})
        _api_call(API_KEY, "GET", "/api/v1/search", params={"q": "test"},
                  base_url=BASE_URL)
        req = mock_urlopen.call_args[0][0]
        self.assertIsNone(req.data)  # GET has no body


if __name__ == "__main__":
    unittest.main()
