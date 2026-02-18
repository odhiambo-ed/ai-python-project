from src.greet import greet


def test_greet():
    assert greet("Ed") == "Hello, Ed! Welcome to AI Engineering."


def test_greet_empty():
    assert greet("") == "Hello, ! Welcome to AI Engineering."
