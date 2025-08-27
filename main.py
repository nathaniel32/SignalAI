from services.data_manager import DataManager
from nn_signal.trainer import Trainer
from nn_signal.predictor import Predictor
import config
import numpy as np

class Main:
    def __init__(self):
        self.data_manager = DataManager()
        self.trainer = Trainer()
        self.menu_items = [
            ('Drop All Tables', lambda: self.data_manager.drop_all_tables()),
            ('Create Tables', lambda: self.data_manager.create_tables()),
            ('Import CSV', lambda: self.import_csv_to_database()),
            ('Train AI', lambda: self.trainer.main(df=self.data_manager.get_data())),
            ('Predict Signal', lambda: self.predict()),
            ('Etoro Data', lambda: self.data_manager.import_json_to_database_etoro(market_id=28)),
        ]

    def import_csv_to_database(self):
        file_path = input("CSV path: ").strip()
        market_id = input("Market ID: ").strip()
        symbol = input("Symbol: ").strip()
        period = input("Period: ").strip()
        self.data_manager.import_csv_to_database(file_path, market_id, symbol, period, date_column="Date", open_column="Open", high_column="High", low_column="Low", close_column="Close", volume_column="Volume", adjusted_close_column="Adjusted Close")
    
    def predict(self):
        predictor = Predictor(config.TRAINED_PATH)
        pred_index, (classes, probs) = predictor.main()
        sorted_idx = np.argsort(probs)[::-1]
        print("Hasil prediksi:")
        for i in sorted_idx:
            print(f"- {classes[i]}\t: {probs[i]*100:.2f}%")
    
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
    app = Main()
    app.data_manager.start_session()
    app.data_manager.create_tables()
    app.menu()
    app.data_manager.close_session()