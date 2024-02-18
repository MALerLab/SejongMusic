from typing import List, Union
import numpy as np
from .constants import MEAS_LEN_BY_IDX

def make_dynamic_template(offset_list=MEAS_LEN_BY_IDX, beat_sampling_num=6):
  whole_dynamic_template = []
  for part_idx in range(8):
      frame = beat_sampling_num
      # if part_idx == 0:
      #     # 세종실록악보 : 2.5/1.5, 1.5/1/1.5
      #     dynamic_mapping = {0.0: 'strong', 2.5: 'strong', 1.5: 'middle', \
      #         0.5: 'weak', 1.0: 'weak', 2.0: 'weak', 3.0: 'weak',3.5: 'weak'}
      #     # frame = frame * 2 #세종실록인 경우는 frame이 두 배!
      
      if part_idx in [0,1]:
          # 금합자보: 5/3, 3/2/3
          dynamic_mapping = {0.0: 'strong', 5.0: 'strong', 3.0: 'middle',\
              1.0: 'weak', 2.0: 'weak', 4.0: 'weak', 6.0: 'weak', 7.0: 'weak',}
      
      elif part_idx in [2, 3, 4, 5]:
          # 속악원보, 금보신증가령, 한금신보, 어은보: 6/4, 3/3/2/2 
          dynamic_mapping = {0.0: 'strong', 6.0: 'strong', 3.0: 'middle', 8.0: 'middle', \
              1.0: 'weak', 2.0: 'weak', 4.0: 'weak', 5.0: 'weak', 7.0: 'weak', 9.0: 'weak'}
      
      elif part_idx in [6, 7]:
          # 삼죽금보, 현행: 5/5, 3/2/2/3
          dynamic_mapping = {0.0: 'strong', 5.0: 'strong', 3.0: 'middle', 7.0: 'middle',\
              1.0: 'weak', 2.0: 'weak', 4.0: 'weak', 6.0: 'weak', 8.0: 'weak', 9.0: 'weak'}
      whole_measure_len = int(frame * offset_list[part_idx])
      dynamic_list = []
      for i in range(whole_measure_len):
          dynamic = dynamic_mapping.get(float(i) / frame, 'none')
          dynamic_list.append(dynamic)
      whole_dynamic_template.append(dynamic_list)
  return whole_dynamic_template

def convert_dynamics_to_integer(dynamics):
  dynamic_mapping = {'strong': 3, 'middle':4 , 'weak':5, 'none': 6}
  return [dynamic_mapping[d] for d in dynamics]


def convert_note_to_sampling_with_str(measure, dynamic_templates, beat_sampling_num=6):
  if len(measure) == 0:
    return [['empty'] for _ in range(60)]  # 어차피, 4,5만 비어있으니까 괜찮을 것 같음! 
    
  new_measure = []
  for note in measure:
    if note[2] == 0:
      continue
    frame = int(beat_sampling_num * note[2])
    if note[0]==0:
      frame = frame * 2 # beat strength는 절대값?
    # [part_idx, pitch, onset]
    new_note = [[note[0], note[1], 1]] + [[note[0], note[1], 0]]*(frame-1)
    new_measure += new_note
      
  dynamic_template = dynamic_templates[new_measure[0][0]] #part_idx
  dynamic_template = dynamic_template * (len(new_measure)//len(dynamic_template))
  new_measure_with_dynamic = []
  # print(len(new_measure), len(dynamic_template))
  for i, frame in enumerate(new_measure):
    if len(new_measure) == len(dynamic_template):
      new_frame = frame+[dynamic_template[i]]
      new_measure_with_dynamic.append(new_frame)
    else:
      ValueError('len(new_measure) != len(dynamic_template)')
  return new_measure_with_dynamic

def convert_note_to_sampling(measure, dynamic_templates, beat_sampling_num = 6):
  if len(measure) == 0:
    return [['empty'] for _ in range(10 * beat_sampling_num)]  # 어차피, 4,5만 비어있으니까 괜찮을 것 같음! 
  
  measure_duration = sum([note[2] for note in measure])
  dynamic_template = dynamic_templates[measure[0][0]] #part_idx
  num_measures = round(measure_duration * beat_sampling_num / len(dynamic_template))
  new_measure = np.zeros((int(num_measures * len(dynamic_template)), 5), dtype=np.int32) # part_idx, pitch, onset, dynamic, position
  new_measure[:, 0] = measure[0][0] # part_idx
  new_measure[:, 3] = dynamic_template * num_measures
  new_measure[:, 4] = np.arange(len(new_measure))
  current_beat = 0


  for note in measure:
    onset_frame = int(beat_sampling_num * current_beat)
    offset_frame = int(beat_sampling_num * (current_beat + note[2]))
    new_measure[onset_frame:offset_frame, 1] = note[1] # pitch
    new_measure[onset_frame, 2] = 1
    current_beat += note[2]

  return new_measure

def convert_onset_to_sustain_token(roll_array: np.ndarray):
  roll_array[roll_array[:,2]!=1, 1] = 0
  return roll_array[:, [0,1,3,4]]


def get_similarity(sampled_notes, idx_1, idx_2):
  similarity = []
  for i in range(160):
    first = sampled_notes[idx_1][i]
    second = sampled_notes[idx_2][i]
    
    if idx_1 in [2,3,4,5,6,7] or idx_2 in [4,5]:
      if first[0]==['empty'] or second[0]==['empty']:
        continue
    measure_similarity = sum([note_1[1]==note_2[1] for note_1, note_2 in zip(first, second)])/len(first)
    similarity.append(measure_similarity)
   
  return sum(similarity)/len(similarity)

def get_correspondence(sampled_notes, idx_1, idx_2):
  correspondence = []
  for i in range(160):
    first = sampled_notes[idx_1][i]
    second = sampled_notes[idx_2][i]
    if idx_1 in [2,3,4,5,6,7] or idx_2 in [4,5]:
      if first[0]==['empty'] or second[0]==['empty']:
        continue
    strong_beat_a = []   
    strong_beat_b = []
    for j, note in enumerate(first):
      if note[3] in ['strong', 'middle']:
        strong_beat_a.append(note[1])
    for j, note in enumerate(second):
      if note[3] in ['strong', 'middle']:
        strong_beat_b.append(note[1])
    
    measure_correspondence = sum([strong_beat_a[idx]==strong_beat_b[idx] for idx in range(len(strong_beat_a))])/len(strong_beat_a)
    correspondence.append(measure_correspondence)
   
  return sum(correspondence)/len(correspondence)


def get_similarity_with_inference(src:List[Union[int, float, str]], output:List[Union[int, float, str]], dynamic_templates):
  sampled_src = convert_note_to_sampling_with_str(src, dynamic_templates)
  smpled_output = convert_note_to_sampling_with_str(output, dynamic_templates)
  
  measure_similarity = sum([note_1[1]==note_2[1] for note_1, note_2 in zip(sampled_src, smpled_output)])/len(sampled_src)

  return measure_similarity

def get_correspondence_with_inference(src, output, dynamic_templates):
  sampled_src = convert_note_to_sampling_with_str(src, dynamic_templates)
  smpled_output = convert_note_to_sampling_with_str(output, dynamic_templates)
  strong_beat_a, strong_beat_b = [], []
  for j, note in enumerate(sampled_src):
    if note[3] in ['strong', 'middle']:
      strong_beat_a.append(note[1])
  for j, note in enumerate(smpled_output):
    if note[3] in ['strong', 'middle']:
      strong_beat_b.append(note[1])
  if len(strong_beat_a) != len(strong_beat_b):
    print("src")
    print(src)
    print("output")
    print(output)
    print(strong_beat_a, strong_beat_b)
  measure_correspondence = sum([strong_beat_a[idx]==strong_beat_b[idx] for idx in range(len(strong_beat_a))])/len(strong_beat_a)
  return measure_correspondence



if __name__ == "__main__":
  from yeominrak_processing import ShiftedAlignedScore

  train_dataset = ShiftedAlignedScore(is_valid = False)

  pass