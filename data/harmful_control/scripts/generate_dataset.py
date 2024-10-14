import csv
import os
import re
from openai import OpenAI
from tqdm import tqdm
import argparse

PROMPT = 'I will provide some keywords that might lead a large language model to exhibit inappropriate behavior. Your task is to help me modify these keywords. Ensure that the modified keywords closely match the original in meaning, while avoiding leading the large language model into inappropriate behavior. Please present the modified keywords in the form of a Python list, using double quotes (e.g., ["str", "str"]), without any additional descriptions. Samples:\n'


class OpenAIChatBot:
    def __init__(self, api_key, base_url, prompt, model='gpt-4-1106-preview', seed=42):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.prompt = prompt
        self.model = model
        self.seed = seed

    def get_response(self, msg, temperature=0, top_p=0.5):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": self.prompt + msg}],
            temperature=temperature,
            top_p=top_p,
            seed=self.seed
        )
        return completion.choices[0].message.content


def read_csv(file_name):
    with open(file_name, newline='') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        return [row for row in csvreader]


def save_csv(data, filename):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def extract_and_convert_list(str_with_list):
    list_str = re.search(r'\[.*?]', str_with_list, re.DOTALL)
    list_str = list_str.group()
    return eval(list_str)


def process_data(bot, data, batch_size):
    results = []
    for idx in tqdm(range(0, len(data), batch_size)):
        msg = str(data[idx: idx + batch_size])
        response = bot.get_response(msg)
        results += extract_and_convert_list(response)
    return results


def generate_and_save(dataset, file_name):

    instructions = [sample[0] for sample in dataset[1:]]

    modified_instructions = process_data(bot, instructions, 20)

    final_result = [('Text', 'Label')] + [(x, 1) for x in instructions] + [(x, 0) for x in modified_instructions]

    save_csv(final_result, file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate dataset")
    parser.add_argument('--api_key', help="OpenAI API key", required=True)
    parser.add_argument('--base_url', help="base url", default='https://api.openai.com/v1')
    parser.add_argument('--gen_num', help='The number of files to be generated', type=int, default=1)
    parser.add_argument('--model', help="model name", default='gpt-4-1106-preview')
    parser.add_argument('--input_csv', help="Input CSV file path", default='../harmful_behaviors.csv')
    parser.add_argument('--output_folder', help="Output CSV file path", default='../')
    args = parser.parse_args()

    bot = OpenAIChatBot(args.api_key, args.base_url, PROMPT, args.model)

    data = read_csv(args.input_csv)

    for i in range(args.gen_num):
        print(f'Generating...  {i + 1} / {args.gen_num}')
        file_name = os.path.join(args.output_folder, 'combined_instructions_' + str(i) + '.csv')
        generate_and_save(data, file_name)
        print(f'{file_name} dataset is saved.')
