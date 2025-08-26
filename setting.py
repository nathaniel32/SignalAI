from services.data_manager import DataManager

class Setting:
    def __init__(self):
        self.data_manager = DataManager()
        self.menu_items = [
            ('Drop All Tables', lambda: self.data_manager.drop_all_tables()),
            ('Create Tables', lambda: self.data_manager.create_tables()),
        ]

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