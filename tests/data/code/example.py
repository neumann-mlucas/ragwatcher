"""Sample code file for reader test."""


class Widget:
    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        return f"hello from {self.name}"


def make_widget(name: str = "default") -> Widget:
    return Widget(name)


if __name__ == "__main__":
    w = make_widget("alpha")
    print(w.greet())
