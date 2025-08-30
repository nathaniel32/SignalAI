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
            ('Train AI', lambda: self.trainer.main(datasets_df=self.data_manager.get_datasets(period=self.data_manager.print_get_period()))),
            ('Predict Signal', lambda: self.predict()),
            ('Etoro Data', lambda: self.import_json_to_database_etoro()),
        ]
    
    def import_json_to_database_etoro(self):
        while True:
            market_id=input("ID: ")
            if market_id:
                self.data_manager.import_json_to_database_etoro(market_id=market_id)
            else:
                break

    def import_csv_to_database(self):
        file_path = input("CSV path*: ").strip()
        market_id = input("Market ID*: ").strip()
        symbol = input("Symbol: ").strip()
        period = input("Period*: ").strip()

        date_column = input("Date column (default: Date): ") or "Date"
        open_column = input("Open column (default: Open): ") or "Open"
        high_column = input("High column (default: High): ") or "High"
        low_column = input("Low column (default: Low): ") or "Low"
        close_column = input("Close column (default: Close): ") or "Close"
        volume_column = input("Volume column (default: Volume): ") or "Volume"
        adjusted_close_column = input("Adjusted Close column (default: Adjusted Close): ") or "Adjusted Close"
        self.data_manager.import_csv_to_database(file_path, market_id, symbol, period, date_column=date_column, open_column=open_column, high_column=high_column, low_column=low_column, close_column=close_column, volume_column=volume_column, adjusted_close_column=adjusted_close_column)
    
    def predict(self):
        predictor = Predictor(config.TRAINED_PATH)
        while True:
            market_id = input("Market ID: ")
            period = self.data_manager.print_get_period()
            
            if market_id and period:
                try:
                    self.data_manager.import_json_to_database_etoro(market_id=market_id, period=period)
                    df = self.data_manager.get_data(market_id=market_id, period=period)
                    
                    (classes, probs), pred_index = predictor.main(df=df, market_id=market_id, period=period)
                    
                    sorted_idx = np.argsort(probs)[::-1]
                    print("\nResults:")
                    for i in sorted_idx:
                        print(f"- {classes[i]}\t: {probs[i]*100:.2f}%")
                except RuntimeError as e:
                    print(e)
                except Exception as e:
                    print("Prediction error:", e)
            else:
                break
    
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