from importlib.metadata import version


def test_package_version_matches_installed_metadata() -> None:
    import mgts_sensing

    assert mgts_sensing.__version__ == version("mgts-fiber-sensing")
