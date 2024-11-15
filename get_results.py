import requests
import argparse
import configparser


class GetResults:
    def __init__(self):
        self.image_file_path = ""
        settings_config = configparser.ConfigParser()
        settings_config.read_file(open(r"config.ini", encoding="utf-8"))
        self.port_num = int(settings_config["API_PARAMS"]["port_num"])

    def get_json_response(self):

        url = f'http://127.0.0.1:{self.port_num}/predict'
        with open(self.image_file_path, 'rb') as image_file:
            files = {'file': image_file}
            try:
                response = requests.post(url, files=files)
                response.raise_for_status()
                result = response.json()
                return result
            except requests.exceptions.RequestException as e:
                print(f"Error calling API: {e}")
                return None

    def get_cli_args(self):
        """
        Gets Command line arguments
        :return:
        """
        cli_parser = argparse.ArgumentParser()
        cli_parser.add_argument("--file_path", help="Path of image to process")
        args = cli_parser.parse_args()
        self.image_file_path = args.file_path

    def trigger_exec(self):
        """
        Triggers execution in a flow
        :return:
        """
        self.get_cli_args()
        print(self.get_json_response())


if __name__ == '__main__':
    get_results_obj = GetResults()
    get_results_obj.trigger_exec()

