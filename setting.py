from database.connection import Connection

class Setting:
    def __init__(self):
        self.session = None
        self.db_connection = Connection()

    def start_session(self):
        self.session = self.db_connection.get_session()

    def close_session(self):
        self.db_connection.close_session()

    def menu(self):
        self.db_connection.create_tables()
        pass

if __name__ == "__main__":
    app = Setting()
    app.start_session()
    app.menu()
    app.close_session()