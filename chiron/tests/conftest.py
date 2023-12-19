import pytest


@pytest.fixture
def remove_h5_file():
    import os

    if os.path.exists("test.h5"):
        os.remove("test.h5")
    yield
    if os.path.exists("test.h5"):
        os.remove("test.h5")
