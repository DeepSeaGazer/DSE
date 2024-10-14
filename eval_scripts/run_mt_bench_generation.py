import argparse
import subprocess
import os


def main():
    parser = argparse.ArgumentParser(description='Run MT-Bench generation.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file')
    args = parser.parse_args()

    os.chdir("../eval")

    coeff_preset = [0, 1, 2, 3]

    for coeff in coeff_preset:
        command = [
            "python", 'mt_bench.py',
            "--model_path", args.model_path,
            "--coeff", str(coeff),
        ]

        subprocess.run(command)


if __name__ == '__main__':
    main()
