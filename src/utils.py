def assert_field_exists(self, attr: str):
    try:
        assert(getattr(self, attr, None) is not None)
    except:
        raise TypeError(f"Attribute {attr} does not exist in {self.__class__.__name__}")