import json
import os
import hashlib

from src import user


class UserManager:
    """
    User manager to register a new user or login
    Passwords are hashed SHA256

    """

    def __init__(self):
        self.users = {}
        if os.path.exists("users.json"):
            with open('users.json', 'r') as f:
                try:
                    self.users = json.load(f)
                except Exception as e:
                    print(f'Can not load the user file, error: {e}')

    def register_new_user(self, username, password):
        if username in self.users:
            return False, f'User {username} already exists'

        self.users[username] = hashlib.sha512(password.encode('utf-8')).hexdigest()
        with open('users.json', 'w') as f:
            json.dump(self.users, f, indent=4)
        return True, 'User created successfully'

    def login(self, username, password):
        if username in self.users:
            if self.users[username] == hashlib.sha512(password.encode('utf-8')).hexdigest():
                return True, 'Login successful'
            else:
                return False, 'Incorrect password'
        else:
            return False, 'User not found'
