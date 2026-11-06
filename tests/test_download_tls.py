"""Regression checks for macOS Python installations without configured roots."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import ssl
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

from streamgraph.cli import main
from streamgraph.data import download_ssl_context, fetch_archive


class VerifiedDownloads(unittest.TestCase):
    def test_context_works_without_default_ca_paths_and_keeps_verification(self):
        empty_default = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        # Some hosted runtimes inject a private trust root even into a bare
        # context. Do not assume that its initial store has exactly zero roots.
        initial_roots = empty_default.cert_store_stats()["x509_ca"]
        with patch("streamgraph.data.ssl.create_default_context", return_value=empty_default):
            context = download_ssl_context()
        self.assertGreater(context.cert_store_stats()["x509_ca"], initial_roots)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)

    def test_certificate_failure_has_actionable_message(self):
        failure = URLError(ssl.SSLCertVerificationError("unable to get local issuer"))
        with tempfile.TemporaryDirectory() as directory:
            with patch("streamgraph.data.urlopen", side_effect=failure) as opener:
                with self.assertRaisesRegex(OSError, "Install Certificates.command"):
                    fetch_archive("https://data.pyg.org/test.zip", Path(directory) / "test.zip")
                context = opener.call_args.kwargs["context"]
                self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
                self.assertTrue(context.check_hostname)

    def test_installed_console_entry_catches_download_errors(self):
        stderr = io.StringIO()
        with patch("streamgraph.cli.run_workflow", side_effect=OSError("Certificate setup needed")):
            with redirect_stderr(stderr), self.assertRaises(SystemExit) as exit_status:
                main(["download"])
        self.assertEqual(exit_status.exception.code, 1)
        self.assertIn("Certificate setup needed", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
