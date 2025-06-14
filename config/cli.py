# cli.py
import argparse
import inspect

class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, **kwargs):
        if self._initialized:
            return

        self._config = {}
        for key, value in kwargs.items():
            setattr(self, key, value)
            self._config[key] = value
        self._initialized = True

    def __getattr__(self, name):
        # Allow access to config attributes directly
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Prevent setting new attributes after initialization, unless it's a private attribute
        if hasattr(self, '_initialized') and self._initialized and not name.startswith('_'):
            raise AttributeError(f"Cannot set new attribute '{name}' after Config is initialized.")
        super().__setattr__(name, value)

    def print_config(self):
        print("--- CLI Configuration ---")
        for key, value in self._config.items():
            if callable(value) and not inspect.isclass(value):
                # For factories, print their string representation or a generic message
                print(f"  {key}: <factory function>")
            else:
                print(f"  {key}: {value}")
        print("-------------------------")

# Example usage (usually done once at the start of your main script)
# config = Config(seed_everything=1337, compile=True)
# config.print_config()
