import torch
from music21 import stream
from .model_zoo import Seq2seq
from .utils import convert_note_to_sampling, convert_onset_to_sustain_token

def sequential_inference(model:Seq2seq, sample:torch.Tensor, decoder):
  s = stream.Score(id='mainScore')
  
  # output_part_idx = 1
  for i in range(7):
    if model.is_condition_shifted:
      src, output, attention_map = model.shifted_inference(sample, i+1)
    else:
      src, output, attention_map = model.inference(sample, i+1)    
    if i == 0:
      merged_part = stream.Part() 
      src_midi = decoder(src)
      for element in src_midi:
        merged_part.append(element)
      s.insert(0, merged_part)
    output_midi = decoder(output)
    merged_part = stream.Part() 
    for element in output_midi:
      merged_part.append(element)
    s.insert(0, merged_part)

    if model.is_condition_shifted:
      pred, condition = [note[:3] for note in output], [note[3:] for note in output]
      initial_condition_token = []
      if 'offset' in model.tokenizer.key_types:
        initial_condition_token.append(0)
      if 'dynamic' in model.tokenizer.key_types:
        initial_condition_token.append('strong')
      if 'measure_idx' in model.tokenizer.key_types:
        initial_condition_token.append(0)
      if 'measure_changed' in model.tokenizer.key_types:
        initial_condition_token.append(1)

      output = [pred[0] + initial_condition_token] + [pr + con for pr, con in zip(pred[1:], condition[:-1])] + [[i+1] + ['end'] * (len(model.tokenizer.key_types)-1)]
      output = [[i+1] +  ['start'] * (len(model.tokenizer.key_types)-1)] + output
      sample = [model.tokenizer(note) for note in output]
    elif hasattr(model, 'sampling_rate'):
      converted_output = convert_note_to_sampling(output, model.dynamic_template_list, beat_sampling_num=model.sampling_rate)
      converted_output = convert_onset_to_sustain_token(converted_output)
      sample = [model.tokenizer(item) for item in converted_output]

    else:
      output = [[i+1] +  ['start'] * (len(model.tokenizer.key_types)-1)] + output + [[i+1] + ['end'] * (len(model.tokenizer.key_types)-1)]
      sample = [model.tokenizer(note) for note in output]

    sample = torch.Tensor(sample).long()

  return s