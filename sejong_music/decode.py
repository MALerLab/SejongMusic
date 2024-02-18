from music21 import pitch as m21_pitch, environment, converter, stream, meter, tempo, key, note as m21_note


class MidiDecoder:
  def __init__(self, tokenizer):
    # self.model = model
    # self.valid_sample_list = [loader.dataset[i][0] for i in range(len(loader.dataset))]
    self.tokenizer = tokenizer
    self.vocab = self.tokenizer.vocab
    # self.converted_predicted_list = self.converted_to_org_value(output)

  def make_stream_obj(self, output):
    stream_obj = stream.Stream()
    
    # meter 만들기
    part_idx = output[0][0]
    if part_idx == 0:
      sig = "8/8"
    elif part_idx == 1:
      sig = "8/4"
    elif part_idx in [2,3,4,5,6,7]:
      sig = "10/4"
    time_signature = meter.TimeSignature(sig)
    current_key = key.KeySignature(-4)
    stream_obj.append(time_signature) 
    stream_obj.append(current_key)
    
    # tempo 만들기
    bpm = 100
    qpm = tempo.MetronomeMark(number=bpm)
    stream_obj.append(qpm)
    
    # note 추가
    notes_data = [(info[1], info[2]) for info in output]
    for pitch, duration in notes_data:
      n = m21_note.Note(pitch)
      n.pitch = m21_pitch.simplifyMultipleEnharmonics(pitches=[n.pitch], keyContext=current_key)[0]
      if n.pitch.accidental.alter == 0:
        n.pitch.accidental = None # delete natural
      if part_idx == 0:
        n.duration.quarterLength = duration/2
      else:
        n.duration.quarterLength = duration
      if n.duration.quarterLength == 0:
        n = n.getGrace()
      stream_obj.append(n)
    return stream_obj
    
  def __call__(self, output):
    stream_obj = self.make_stream_obj(output)
    return stream_obj

class OrchestraDecoder:
  def __init__(self, tokenizer):
    self.tokenizer = tokenizer
    self.vocab = self.tokenizer.vocab

  def make_stream_obj(self, output):
    stream_obj = stream.Stream()
    
    # meter 만들기
    part_idx = output[0][0]
    sig = "60/8"
    time_signature = meter.TimeSignature(sig)
    current_key = key.KeySignature(-4)
    stream_obj.append(time_signature) 
    stream_obj.append(current_key)
    
    # tempo 만들기
    bpm = 100
    qpm = tempo.MetronomeMark(number=bpm)
    stream_obj.append(qpm)
    
    # note 추가
    notes_data = [(info[1], info[2]) for info in output]
    for pitch, duration in notes_data:
      n = m21_note.Note(pitch)
      n.pitch = m21_pitch.simplifyMultipleEnharmonics(pitches=[n.pitch], keyContext=current_key)[0]
      if n.pitch.accidental.alter == 0:
        n.pitch.accidental = None # delete natural
      n.duration.quarterLength = duration
      if n.duration.quarterLength == 0:
        n = n.getGrace()
      stream_obj.append(n)
    return stream_obj
    
  def __call__(self, output):
    stream_obj = self.make_stream_obj(output)
    return stream_obj
  

class SampingDecoder(MidiDecoder):
  def __init__(self, model, loader):
    super().__init__(model, loader)
    self.beat_sampling_num = loader.dataset.beat_sampling_num
    
  def make_note_value(self, output):
    decoded_notes = []
    previous_pitch_set = [output[0][1], 1]
    
    for i, pair in enumerate(output[1:], start=1):
      part_idx, pitch, is_onset = pair[0], pair[1], pair[2]
      if is_onset == 1: #처음에는 무조건 is_onset = True
        decoded_notes.append(previous_pitch_set)
        previous_pitch_set = [pitch, 1]
      # elif previous_pitch_set[0] != pitch: # 다른 음일 때
      #   decoded_notes.append(previous_pitch_set)
      #   previous_pitch_set = [pitch, 1]
      else:
        # 같은 음일 때
        previous_pitch_set[1] += 1
    decoded_notes.append(previous_pitch_set) # 마지막 음
    decoded_notes = [[part_idx, note[0], note[1]/self.beat_sampling_num] for note in decoded_notes]
    return decoded_notes
  
  def __call__(self, output):
    note_value = self.make_note_value(output)
    stream_obj = self.make_stream_obj(note_value)
    return stream_obj  
