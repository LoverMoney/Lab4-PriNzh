from lab4 import neuro
from PIL import Image
import pytest

dataX = "black.jpg"

@pytest.fixture
def mock_data():
    return dataX

def test_count_men_over_age(mock_data):
    res=neuro(dataX)
    assert res[0]=="a close up of a black square with a white border"
