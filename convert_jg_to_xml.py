from pathlib import Path
from sejong_music.jg_to_staff_converter import JGToStaffConverter

# text_fn = 'gen_results/chwipunghyeong_bert_orchestration_gencode4.txt'

if __name__ == "__main__":
  txt_dir = Path('music_score/jg_dataset')
  

  for txt_fn in txt_dir.glob('*.txt'):
    gen_str = open(txt_fn, 'r').read()
    print(txt_fn)
    converter = JGToStaffConverter()
    entire_notes, entire_stream = converter.convert_multi_track(gen_str, time_signature='30/8')

    entire_stream.write('musicxml', txt_fn.parent / (txt_fn.stem + '.musicxml'))
