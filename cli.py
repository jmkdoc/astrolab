# cli.py
import argparse
import inspect

class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # Directly set internal attributes using __dict__ to avoid __setattr__ recursion
            cls._instance.__dict__['_initialized'] = False
            cls._instance.__dict__['_config'] = {} # Initialize _config here as well
        return cls._instance

    def __init__(self, **kwargs):
        # Access internal attribute directly to avoid __getattr__ recursion
        if self.__dict__['_initialized']:
            return

        # self.__dict__['_config'] is already initialized in __new__
        
        for key, value in kwargs.items():
            # Use super().__setattr__ to avoid custom __setattr__ for initial setup
            super().__setattr__(key, value) 
            # Store in the internal config dictionary using direct __dict__ access
            self.__dict__['_config'][key] = value

        # Mark as initialized using direct __dict__ access
        self.__dict__['_initialized'] = True

    def __getattr__(self, name):
        # Check if attribute is in the internal config dictionary directly
        if name in self.__dict__['_config']:
            return self.__dict__['_config'][name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        # Access internal attributes using __dict__.get for safety and to avoid recursion
        # If '_initialized' is not yet in __dict__, get() returns False, preventing early checks.
        is_initialized = self.__dict__.get('_initialized', False)

        # Prevent setting new attributes after initialization, unless it's a private attribute (starts with '_')
        if is_initialized and not name.startswith('_'):
            raise AttributeError(f"Cannot set new attribute '{name}' after Config is initialized.")
        
        # Always use super().__setattr__ to actually set the attribute in the instance's dictionary
        super().__setattr__(name, value)
        
        # If the object is initialized and it's a public attribute, also update _config
        # This is for attributes set *after* the initial __init__ call
        if is_initialized and not name.startswith('_'):
             self.__dict__['_config'][name] = value


    def print_config(self):
        print("--- CLI Configuration ---")
        # Access _config directly
        for key, value in self.__dict__['_config'].items():
            if callable(value) and not inspect.isclass(value):
                print(f"  {key}: <factory function>")
            else:
                print(f"  {key}: {value}")
        print("-------------------------")