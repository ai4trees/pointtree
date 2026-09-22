"""Tests for pointtree.operations.no_stdout."""

import io
import sys

from pointtree.operations._no_stdout import no_stdout


class TestNoStdout:
    """Tests for pointtree.operations.no_stdout."""

    def test_suppresses_output_with_file_descriptor(self, capfd):
        with no_stdout():
            print("should not appear")
        print("should appear")

        captured = capfd.readouterr()

        assert "should not appear" not in captured.out
        assert "should appear" in captured.out

    def test_suppresses_output_without_file_descriptor(self, monkeypatch):
        # io.StringIO does not provide a real file descriptor, which mimics environments such as Jupyter
        # notebooks where sys.stdout.fileno() raises io.UnsupportedOperation.
        fake_stdout = io.StringIO()
        monkeypatch.setattr(sys, "stdout", fake_stdout)

        with no_stdout():
            print("should not appear")

        assert fake_stdout.getvalue() == ""

    def test_suppresses_output_when_fileno_is_missing(self, monkeypatch):
        class StdoutWithoutFileno:  # pylint: disable=missing-class-docstring
            def write(self, _text):
                pass

            def flush(self):
                pass

        fake_stdout = StdoutWithoutFileno()
        monkeypatch.setattr(sys, "stdout", fake_stdout)

        with no_stdout():
            print("should not raise")
