def pytest_addoption(parser):
    parser.addoption(
        "--address",
        action="store",
        default=None,
        help="The IP address of the Ray head.  Format should be IP:PORT",
    )

    parser.addoption(
        "--tmpdir",
        action="store",
        default="/tmp/ray",
        help="Temporary directory to write files to.",
    )
