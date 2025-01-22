import sys
import os
import argparse
import subprocess

sys.path.append('/home/renkow/arcadia')

import yt.wrapper as yt
from voicetech.tts.training.utils.yt.local_dataset import load_audio, save_audio
from tqdm import tqdm

def create_lst_from_rows(rows, output_lst_path, prompt_audio_dir, ref_audio_dir):
    os.makedirs(prompt_audio_dir, exist_ok=True)
    os.makedirs(ref_audio_dir, exist_ok=True)
    if os.path.dirname(output_lst_path):
        os.makedirs(os.path.dirname(output_lst_path), exist_ok=True)

    lst_lines = []

    for index, row in tqdm(enumerate(rows), desc="Processing rows"):
        prompt_filename = f"prompt_audio_{index}.wav"
        ref_filename = f"ref_audio_{index}.wav"

        prompt_wav_path = os.path.join(prompt_audio_dir, prompt_filename)
        ref_wav_path = os.path.join(ref_audio_dir, ref_filename)

        if 'prompt__wav' in row:
            wav, sr = load_audio(row['prompt__wav'])
            audio_bytes = save_audio(wav, sr, format="WAV", subtype="PCM_16")
            with open(prompt_wav_path, "wb") as f:
                f.write(audio_bytes)

        if 'synthesized__wav' in row:
            wav, sr = load_audio(row['synthesized__wav'])
            audio_bytes = save_audio(wav, sr, format="WAV", subtype="PCM_16")
            with open(ref_wav_path, "wb") as f:
                f.write(audio_bytes)

        prompt_text = row.get('prompt_text', '').strip().replace('|', ' ').replace('\n', ' ')
        synthesized_text = row.get('text', '').strip().replace('|', ' ').replace('\n', ' ')

        line = f"{ref_filename}|{prompt_text}|{os.path.join('prompt-wavs', prompt_filename)}|{synthesized_text}"
        lst_lines.append(line)

    with open(output_lst_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lst_lines))

    print(f".lst file saved at {output_lst_path}")

def run_command(command):
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, text=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        sys.exit(1)

def calculate_sim_and_wer(meta_lst, ref_audio_dir, sim_checkpoint):
    sim_script = "/home/renkow/seed-tts-eval/cal_sim.sh"
    wer_script = "/home/renkow/seed-tts-eval/cal_wer.sh"

    sim_command = f"bash {sim_script} {meta_lst} {ref_audio_dir} {sim_checkpoint}"
    run_command(sim_command)

    wer_command = f"bash {wer_script} {meta_lst} {ref_audio_dir} en"
    run_command(wer_command)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yt-path", required=True)
    parser.add_argument("--output-lst", required=True)
    parser.add_argument("--prompt-audio-dir", required=True)
    parser.add_argument("--ref-audio-dir", required=True)
    parser.add_argument("--sim-checkpoint", required=True)

    args = parser.parse_args()

    client = yt.YtClient(proxy="hahn")
    rows = client.read_table(yt.TablePath(args.yt_path))

    create_lst_from_rows(
        rows=rows,
        output_lst_path=args.output_lst,
        prompt_audio_dir=args.prompt_audio_dir,
        ref_audio_dir=args.ref_audio_dir
    )

    calculate_sim_and_wer(
        meta_lst=args.output_lst,
        ref_audio_dir=args.ref_audio_dir,
        sim_checkpoint=args.sim_checkpoint
    )

if __name__ == "__main__":
    main()
