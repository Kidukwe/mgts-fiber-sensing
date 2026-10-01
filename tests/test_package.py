def test_package_is_importable() -> None:
    import mgts_sensing

    assert mgts_sensing.__version__ == "0.1.0"
