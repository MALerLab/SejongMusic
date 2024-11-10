from tqdm.auto import tqdm
from sejong_music.era_inference import EraTransformer
from sejong_music.jg_code import JeongganPiece

DEVICE = 'cuda'
INPUT_FN = 'music_score/chwipunghyeong_gen.txt'
TARGET_INST = 'piri'
OUTPUT_FN = 'music_score/chwipunghyeong_bert_gen_test.txt'
NUM_JG_PER_GAK = 10



if __name__ == "__main__":
  era_transformer = EraTransformer('models/bert_checkpoints', device=DEVICE)
  gen_str = open(INPUT_FN).read()

  inst = TARGET_INST
  idx = 0
  self = era_transformer
  total_gen_str = ''

  num_frames_per_measure = 6 * NUM_JG_PER_GAK
  num_use_measure = 2
  num_use_frames = num_frames_per_measure * num_use_measure

  piece = JeongganPiece(None, gen_str=gen_str, inst_list=[inst])
  x = piece.convert_tokens_to_roll(piece.sliced_parts_by_measure[idx][inst], inst)
  x = self.convert_to_input_format(x, add_start_end=False)
  cond = self.make_dummy_condition_tensor()
  x, loss_mask = self.masker.mask_except_note_onset_to_ten(x, cond)
  # x, loss_mask = self.aug_form_to_input_form(x), self.aug_form_to_input_form(loss_mask)
  x = self.unmask_pitch_and_ornaments(x.to(self.device), loss_mask.to(self.device), 2)

  prev_x = x[1+num_frames_per_measure:1+num_use_frames+num_frames_per_measure]
  new_x = x[1:1+num_use_frames+num_frames_per_measure]
  roll = self.tokenizer.decode(new_x)
  omr_str = self.roll2omr(roll)
  new_gen_str = self.omr2gen.convert_lines_to_gencode(omr_str.split('\n')) + ' \n '
  total_gen_str += new_gen_str
  for idx in tqdm(range(1, len(piece.sliced_parts_by_measure))):
    # for idx in tqdm(range(1, 10)):
    x = piece.convert_tokens_to_roll(piece.sliced_parts_by_measure[idx][inst], inst)
    x = self.convert_to_input_format(x, add_start_end=False)
    cond = self.make_dummy_condition_tensor()
    x, loss_mask = self.masker.mask_except_note_onset_to_ten(x, cond)
    x[1:1+num_use_frames, :2] = prev_x[:, :2]
    loss_mask[1:1+num_use_frames, :2] = 0
    # x, loss_mask = self.aug_form_to_input_form(x), self.aug_form_to_input_form(loss_mask)
    x = self.unmask_pitch_and_ornaments(x.to(self.device), loss_mask.to(self.device), 2)

    num_use_frames = num_frames_per_measure * num_use_measure
    prev_x = x[1+num_frames_per_measure:1+num_frames_per_measure+num_use_frames]
    if idx == len(piece.sliced_parts_by_measure) - 1:
      new_x = x[1+num_use_frames:-1]
    else:
      new_x = x[1+num_use_frames:1+num_use_frames+num_frames_per_measure]
    # total_outs.append(new_x)
    roll = self.tokenizer.decode(new_x)
    omr_str = self.roll2omr(roll)
    new_gen_str = self.omr2gen.convert_lines_to_gencode(omr_str.split('\n')) + ' \n '
    total_gen_str += new_gen_str

  with open(OUTPUT_FN, 'w') as f:
    f.write(total_gen_str)

  notes, score = self.gen2staff(total_gen_str, time_signature=f'{NUM_JG_PER_GAK * 3}/8')
  score.write('musicxml', 'cph_bert.musicxml')