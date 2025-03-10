import json
# data["data"][0:n]
# {'dependency': 'click', 'old_version': '==3.0', 'new_version': '==8.0.0', 'old_time': '2014-08-12', 'new_time': '2021-05-11', 'description': 'This Python code defines a command line interface (CLI) using the Click library. It includes a group command with an option for input, which defaults to 23. The CLI returns the value 42. There is also a result callback function that adds the input value to the result and returns the sum.', 'old_code': "@click.group()\n@click.option('-i', '--input', default=23)\ndef cli(input):\n    return 42\n\n@cli.resultcallback()\ndef process_result(result, input):\n    return result + input", 'new_code': "@click.group()\n@click.option('-i', '--input', default=23)\ndef cli(input):\n    return 42\n\n@cli.result_callback()\ndef process_result(result, input):\n    return result + input", 'old_name': 'resultcallback', 'new_name': 'result_callback', 'type': 'name_change', 'edit_order': 'old_to_new', 'language': 'python', 'task': 'code_editing', 'source': 'library_source_code', 'id': 'old_to_new_1'}
if __name__ == "__main__":
    with open("benchmark/data/VersiCode_Benchmark/code_editing/code_editing_new_to_old.json", "r") as f:
        data = json.load(f)

    # print(data)
    for data_piece in data["data"]:
        if data_piece["language"] == "java":
            print(data_piece)
            break