import socket

import pytest


def test_network_is_blocked_during_tests() -> None:
    with pytest.raises(RuntimeError, match="offline"):
        socket.create_connection(("example.com", 80), timeout=1)
