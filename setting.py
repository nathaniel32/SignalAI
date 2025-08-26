from services.data_manager import DataManager

class Setting:
    def __init__(self):
        self.data_manager = DataManager()
        self.menu_items = [
            ('Drop All Tables', lambda: self.data_manager.drop_all_tables()),
            ('Create Tables', lambda: self.data_manager.create_tables()),
            ('Import CSV', lambda: self.import_csv_to_database()),
        ]

    def import_csv_to_database(self):
        file_path = input("CSV path: ").strip()
        market_id = input("Market ID: ").strip()
        symbol = input("Symbol: ").strip()
        period = input("Period: ").strip()
        self.data_manager.import_csv_to_database(file_path, market_id, symbol, period, date_column="Date", open_column="Open", high_column="High", low_column="Low", close_column="Close", volume_column="Volume", adjusted_close_column="Adjusted Close")

    def menu(self):
        while True:
            print("\n=== Menu ===")
            for i, (label, _) in enumerate(self.menu_items):
                print(f"{i+1}. {label}")
            print(f"{len(self.menu_items)+1}. Break")

            choice = input("Input: ").strip()
            print()

            if choice.isdigit():
                choice_num = int(choice)
                if 1 <= choice_num <= len(self.menu_items):
                    self.menu_items[choice_num-1][1]()
                elif choice_num == len(self.menu_items)+1:
                    break
                else:
                    print("Invalid input!")
            else:
                print("Invalid input!")

if __name__ == "__main__":
    app = Setting()
    app.data_manager.start_session()
    app.data_manager.create_tables()
    app.menu()
    app.data_manager.close_session()