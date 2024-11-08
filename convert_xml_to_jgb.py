import argparse
import music21

from sejong_music.jeonggan_utils import JGConverter, GencodeConverter
from pathlib import Path



def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--xml_path', type=str, required=True)
  parser.add_argument('--output_path', type=str, required=True)
  parser.add_argument('--jeonggan_quarter_length', type=float, default=1.5)
  parser.add_argument('--num_jeonggan_per_gak', type=int, default=20)
  return parser.parse_args()


if __name__ == "__main__":
  args = get_args()
  score = music21.converter.parse(args.xml_path)
  
  converter = JGConverter(args.jeonggan_quarter_length, args.num_jeonggan_per_gak)
  converted = converter(score)

  with open(args.output_path, 'w') as f:
    f.write('\n\n'.join(converted))
    
  gen_str = GencodeConverter.convert_txt_to_gencode(args.output_path)
  out_path = Path(args.output_path)
  
  with open(out_path.parent / (out_path.stem + '_gen.txt'), 'w') as f:
    f.write(gen_str)

  print(f"Converted jeonggan is saved at {args.output_path}")