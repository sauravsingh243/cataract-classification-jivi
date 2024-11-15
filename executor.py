import argparse
from train_model import TrainModel
from app import API
import configparser


class Executor:
    def __init__(self):
        self.flag_training = True
        self.train_model_obj = TrainModel()
        self.api_obj = API()
        settings_config = configparser.ConfigParser()
        settings_config.read_file(open(r"config.ini", encoding="utf-8"))
        self.port_num = int(settings_config["API_PARAMS"]["port_num"])

    def trigger_training_flow(self):
        # self.train_model_obj.get_train_test_dataset()
        print("running setup_data_augmentation")
        datagen = self.train_model_obj.setup_data_augmentation()
        print("running load_dataset")
        train_gen, val_gen = self.train_model_obj.load_dataset(datagen)
        print("running build_model")
        built_model = self.train_model_obj.build_model()
        print("running train_model")
        history_data = self.train_model_obj.train_model(built_model, train_gen, val_gen)
        print("running plot_graphs")
        self.train_model_obj.plot_graphs(history_data, built_model, val_gen)
        print("running api")
        self.api_obj.app.run(debug=False, port=self.port_num)

    def trigger_no_training_flow(self):
        # self.train_model_obj.get_train_test_dataset()
        # self.train_model_obj.get_model()
        print("running api")
        self.api_obj.app.run(debug=False, port=self.port_num)

    def trigger_exec(self):
        """
        Triggers execution
        Calls methods in a flow from train_model and app
        :return:
        """
        self.get_cli_args()
        if eval(self.flag_training):
            self.trigger_training_flow()
        else:
            self.trigger_no_training_flow()

    def get_cli_args(self):
        """
        Gets Command line arguments
        :return:
        """
        cli_parser = argparse.ArgumentParser()
        cli_parser.add_argument("--flag_training", help="Specify training or not")
        args = cli_parser.parse_args()
        self.flag_training = args.flag_training


if __name__ == "__main__":
    exec_obj = Executor()
    exec_obj.trigger_exec()


