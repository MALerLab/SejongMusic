from sejong_music import AlignedScore

class TestScore(AlignedScore):
    def __init__(self, xml_path='0_edited.musicxml', 
                 valid_measure_num=[i for i in range(93, 104)], 
                 slice_measure_num=2, 
                 is_valid=False, 
                 use_pitch_modification=False, 
                 pitch_modification_ratio=0, 
                 min_meas=3, 
                 max_meas=6, 
                 transpose=0, 
                 feature_types=['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_idx'],
                 sampling_rate=None) -> None:
        # DYNAMIC_MAPPING[0] = {0.0: 'strong', 1.5: 'strong', 3.0: 'middle'}
        super().__init__(xml_path, valid_measure_num, slice_measure_num, is_valid, use_pitch_modification, pitch_modification_ratio, min_meas, max_meas, feature_types)
        self.transpose_value = transpose
        self.transpose()
        self.result_pairs = self.result_pairs[:len(self.slice_info)]

    # def fix_part_zero_duration(self):
    #     return
    def fix_part_by_rule(self):
        flattened_notes = [SimpleNote(note) for measure in self.parts[0].measures for note in measure]
                  
        measure_shifted_notes = []
        for note in flattened_notes:
          new_offset = (note.measure_offset - 1.5 )
          new_note = None
          if note.pitch == 44:
            note.pitch = 45 # 44 is error
          if note.duration > 4:
            break
          if new_offset < 0:
            new_offset += 4
            note.measure_number -= 1
            note.end_measure_number -= 1
            if note.measure_number <= 0:
              new_note = copy.copy(note)
              new_note.measure_offset = 0
              new_note.duration = 4 - note.duration
              measure_shifted_notes.append(new_note)
              new_note = None
          note.measure_offset = new_offset
          if note.measure_offset + note.duration > 4:
            # print("note duration is too long", note.measure_offset, note.duration)
            extended_duration = note.measure_offset + note.duration - 4
            note.duration -= extended_duration
            new_note = copy.copy(note)
            new_note.measure_offset = 0
            new_note.measure_number += 1
            new_note.end_measure_number += 1
            new_note.duration = extended_duration
          measure_shifted_notes.append(note)
          if new_note:
            measure_shifted_notes.append(new_note)
        new_note = copy.copy(note)
        new_note.measure_offset += note.duration
        new_note.duration = 1.5
        measure_shifted_notes.append(new_note)

        measure_shifted_notes.sort(key=lambda x: (x.measure_number, x.measure_offset))

        # make measure list
        note_by_measure = []
        temp_measure = []
        prev_measure_number = 0
        for note in measure_shifted_notes:
          if note.measure_number != prev_measure_number:
            note_by_measure.append(temp_measure)
            temp_measure = []
            prev_measure_number = note.measure_number
          temp_measure.append(note)
        note_by_measure.append(temp_measure)

        self.parts[0].measures = note_by_measure
        self.measure_features = [self.get_feature(i) for i in range(len(self.parts))]

        part_zero = self.measure_features[0].copy()
        for measure in part_zero:
          for note in measure:
            note[2] *= 2
            note[3] *= 2
        self.measure_features[0] = part_zero
        self.offset_list[0] *= 2


    def transpose(self):
      for idx in range(len(self.parts)):
        part_zero = self.measure_features[idx].copy()
        for measure in part_zero:
          for note in measure:
            note[1] += self.transpose_value
        self.measure_features[idx] = part_zero

    def get_processed_feature(self, front_part_idx, back_part_idx, idx):
      source_start_token = [front_part_idx, 'start', 'start', 'start', 'start', 'start']
      source_end_token = [front_part_idx, 'end', 'end', 'end', 'end', 'end']
      if self.is_valid:
          measure_list = idx
      else:    
          measure_list = self.slice_info[idx]
      original_source_list = [item for idx in measure_list for item in self.measure_features[front_part_idx][idx]]
      if 'measure_idx' in self.tokenizer.key_types:
        m_idx_pos = self.tokenizer.key2idx['measure_idx']
        source_first_measure_idx = original_source_list[0][m_idx_pos]

        original_source_list = [note[:m_idx_pos] + [note[m_idx_pos]-source_first_measure_idx] + note[m_idx_pos+1:] for note in original_source_list]

      source_list = [source_start_token] + original_source_list + [source_end_token]
      source = [self.tokenizer(note_feature) for note_feature in source_list]
      
      return torch.LongTensor(source)
        
    def __getitem__(self, idx):

      front_part_idx, back_part_idx, measure_idx = self.result_pairs[idx]
      src = self.get_processed_feature(front_part_idx, back_part_idx, measure_idx)
      
      return src

class TestScoreCPH(TestScore):
  def __init__(self, xml_path='0_edited.musicxml', valid_measure_num=..., slice_measure_num=2, is_valid=False, use_pitch_modification=False, pitch_modification_ratio=0, min_meas=3, max_meas=6, transpose=0, feature_types=..., sampling_rate=None) -> None:
    super().__init__(xml_path, valid_measure_num, slice_measure_num, is_valid, use_pitch_modification, pitch_modification_ratio, min_meas, max_meas, transpose, feature_types, sampling_rate)


  def fix_part_by_rule(self):
    flattened_notes = [SimpleNote(note) for measure in self.parts[0].measures for note in measure]
              
    measure_shifted_notes = []
    for note in flattened_notes:
      new_offset = (note.measure_offset - 2.5 )
      new_note = None

      ### pitch modification
      if note.pitch == 51:
        note.pitch = 52 # 51 is error
      if note.pitch == 39:
        note.pitch = 40
      if note.pitch == 56:
        note.pitch = 57
      if note.pitch == 57:
        note.pitch = 45

      if new_offset < 0:
        new_offset += 4
        note.measure_number -= 1
        note.end_measure_number -= 1
      note.measure_offset = new_offset
      if note.measure_offset + note.duration > 4:
        # print("note duration is too long", note.measure_offset, note.duration)
        extended_duration = note.measure_offset + note.duration - 4
        note.duration -= extended_duration
        new_note = copy.copy(note)
        new_note.measure_offset = 0
        new_note.measure_number += 1
        new_note.end_measure_number += 1
        new_note.duration = extended_duration
      measure_shifted_notes.append(note)
      if new_note:
        measure_shifted_notes.append(new_note)
    new_note = copy.copy(note)
    new_note.measure_offset += note.duration
    new_note.duration = 2.5
    measure_shifted_notes.append(new_note)

    beginning_dummy_note = copy.copy(measure_shifted_notes[0])
    beginning_dummy_note.measure_offset = 0
    beginning_dummy_note.duration = 1.5
    measure_shifted_notes.append(beginning_dummy_note)

    measure_shifted_notes.sort(key=lambda x: (x.measure_number, x.measure_offset))

    # make measure list
    note_by_measure = []
    temp_measure = []
    prev_measure_number = 0
    for note in measure_shifted_notes:
      if note.measure_number != prev_measure_number:
        note_by_measure.append(temp_measure)
        temp_measure = []
        prev_measure_number = note.measure_number
      temp_measure.append(note)
    note_by_measure.append(temp_measure)

    self.parts[0].measures = note_by_measure
    self.measure_features = [self.get_feature(i) for i in range(len(self.parts))]

    part_zero = self.measure_features[0].copy()
    for measure in part_zero:
      for note in measure:
        note[2] *= 2
        note[3] *= 2
    self.measure_features[0] = part_zero
    self.offset_list[0] *= 2



# ------------ Sampling-based --------------- #
class SamplingTokenizer(Tokenizer):
  def __init__(self, parts, feature_types=['index', 'pitch', 'dynamic',  'offset']):
    num_parts = len(parts)
    self.key_types = feature_types
    vocab_list = defaultdict(list)
    vocab_list['index'] = [i for i in range(num_parts)]
    vocab_list['pitch'] = ['pad', 'start', 'end', 0] + sorted(list(set([note.pitch for i in range(num_parts) for note in parts[i].tie_cleaned_notes])))
    if 'dynamic' in feature_types:
        vocab_list['dynamic'] = ['pad', 'start', 'end'] + list(range(7))
    if 'offset' in feature_types:
        vocab_list['offset'] = ['pad', 'start', 'end'] + list(range(240))

    self.measure_duration = [parts[i].measure_duration for i in range(num_parts)]
    self.vocab = vocab_list
    self.vocab_size_dict = {key: len(value) for key, value in self.vocab.items()}
    self.tok2idx = {key: {k:i for i, k in enumerate(value)} for key, value in self.vocab.items() }
    self.key2idx = {key: i for i, key in enumerate(self.key_types)}
    # self.key_types = ['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_change']

    self.note2token = {}

class ModifiedTokenizer:
    def __init__(self, parts):
        num_parts = len(parts)
        vocab_list = defaultdict(list)
        vocab_list['index'] = ['pad']+[i for i in range(num_parts)]
        vocab_list['pitch'] = ['pad'] + sorted(list(set([note.pitch for i in range(num_parts) for note in parts[i].tie_cleaned_notes])))
        # vocab_list['is_onset'] = ['pad', 'start', 'end'] + [0, 1]
        vocab_list['is_onset'] = [0, 1]
        vocab_list['dynamic'] = ['pad'] + ['strong', 'middle', 'weak']
        
        self.measure_duration = [parts[i].measure_duration for i in range(num_parts)]
        self.vocab = vocab_list
        self.vocab_size_dict = {key: len(value) for key, value in self.vocab.items()}
        self.tok2idx = {key: {k:i for i, k in enumerate(value)} for key, value in self.vocab.items() }

    def __call__(self, note_feature:list):
        key_types = ['index', 'pitch', 'is_onset', 'dynamic']
        converted_lists = [self.tok2idx[key_types[i]][element] for i, element in enumerate(note_feature)]
        
        return converted_lists


# ---------- Sampling-based ----------- #
class SamplingScore(AlignedScore):
  def __init__(self, xml_path='0_edited.musicxml', 
               valid_measure_num=[i for i in range(93, 104)], 
               slice_measure_num=2, is_valid=False, 
               use_pitch_modification=False, 
               pitch_modification_ratio=0, 
               min_meas=3, 
               max_meas=6, 
               feature_types=[],
               sampling_rate=2) -> None:
    super().__init__(xml_path, valid_measure_num, slice_measure_num, is_valid, use_pitch_modification, pitch_modification_ratio, min_meas, max_meas, feature_types)
    self.sampling_rate = sampling_rate
    self.dynamic_template_list = make_dynamic_template(beat_sampling_num=self.sampling_rate)
    self.dynamic_template_list = [convert_dynamics_to_integer(d) for d in self.dynamic_template_list]
    self.tokenizer = SamplingTokenizer(self.parts)

  def get_processed_feature(self, front_part_idx, back_part_idx, idx):
    if self.is_valid:
        measure_list = idx
    else:    
        measure_list = self.slice_info[idx]

    for i, idx in enumerate(measure_list):
        if len(self.measure_features[front_part_idx][idx]) == 0:
            return torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([])
        if len(self.measure_features[back_part_idx][idx]) == 0:
            return torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([])


    original_source_list = [item for idx in measure_list for item in self.measure_features[front_part_idx][idx]]
    original_target_list = [item for idx in measure_list for item in self.measure_features[back_part_idx][idx]]


    source_roll = convert_note_to_sampling(original_source_list, self.dynamic_template_list, beat_sampling_num=self.sampling_rate)
    target_roll = convert_note_to_sampling(original_target_list, self.dynamic_template_list, beat_sampling_num=self.sampling_rate)

    source_roll = convert_onset_to_sustain_token(source_roll)
    target_roll = convert_onset_to_sustain_token(target_roll)

    enc_in = [self.tokenizer(item) for item in source_roll]
    dec_in = [self.tokenizer(item) for item in target_roll]

    enc_in = torch.tensor(enc_in, dtype=torch.long)
    dec_in = torch.tensor(dec_in, dtype=torch.long)

    target = dec_in[:, 0:2]
    dec_in = dec_in[:, [0, 2, 3]]

    return enc_in, dec_in, target

  def __getitem__(self, idx):
    if self.is_valid:
      front_part_idx, back_part_idx, measure_idx = self.result_pairs[idx]
      src, tgt, shifted_tgt = self.get_processed_feature(front_part_idx, back_part_idx, measure_idx)
      return src, tgt, shifted_tgt
    else:    
      sample_success = False
      while not sample_success:
        front_part_idx = random.choice(range(len(self.parts)-1))
        # back_part_idx should be bigger than front_part_idx
        back_part_idx = random.randint(front_part_idx + 1, min(len(self.parts) - 1, front_part_idx + 2))
        src, tgt, shifted_tgt = self.get_processed_feature(front_part_idx, back_part_idx, idx)          
        if len(src) > 0 and len(tgt) > 0:
          sample_success = True
    return src, tgt, shifted_tgt




class SamplingTestScore(TestScore, SamplingScore):
  def __init__(self, xml_path='0_edited.musicxml', valid_measure_num=[i for i in range(93, 104)], slice_measure_num=2, 
               is_valid=False, use_pitch_modification=False, pitch_modification_ratio=0, min_meas=3, max_meas=6, transpose=0, 
               feature_types=['index', 'pitch', 'duration', 'offset', 'dynamic', 'measure_idx'],
               sampling_rate=2) -> None:
    TestScore.__init__(xml_path, valid_measure_num, slice_measure_num, is_valid, use_pitch_modification, pitch_modification_ratio, min_meas, max_meas, transpose, feature_types)
    self.sampling_rate = sampling_rate
    self.dynamic_template_list = make_dynamic_template(beat_sampling_num=self.sampling_rate)
    self.dynamic_template_list = [convert_dynamics_to_integer(d) for d in self.dynamic_template_list]
    self.tokenizer = SamplingTokenizer(self.parts)

  def get_processed_feature(self, front_part_idx, back_part_idx, idx):
    return SamplingScore.get_processed_feature(self, front_part_idx, back_part_idx, idx)

  def __getitem__(self, idx):
    return SamplingScore.__getitem__(self, idx)
       

class SamplingScoreOld(AlignedScore):
    def __init__(self, xml_path='/home/danbi/userdata/DANBI/gugakwon/Yeominrak/0_edited.musicxml', \
        slice_measure_num = 4, sample_len = 1000, is_valid=False, use_pitch_modification=False, beat_sampling_num=6) -> None:
        super().__init__(xml_path, slice_measure_num, sample_len, is_valid, use_pitch_modification)
        self.beat_sampling_num = beat_sampling_num
        self.tokenizer = SamplingTokenizer(self.parts)
        self.dynamic_templates = self.make_dynamic_template()
        self.tokenized_dynamics = []
        for tem in self.dynamic_templates:
            temp = [self.tokenizer.tok2idx['dynamic'][i] for i in tem]
            self.tokenized_dynamics.append(temp)
        
        self.converted_features = []
        for features in self.sliced_features:
            converted_features = []
            for feature in features:
                converted_feature = self.convert_to_sampling(feature)
                converted_features.append(converted_feature)
            self.converted_features.append(converted_features)

    def __len__(self):
        if self.is_valid:
            return len(self.valid_index)
        return self.sample_len
    
    def convert_to_sampling(self, data):
        new_data = []
        for measure in data:
            new_measure = []
            for note in measure:
                frame = int(self.beat_sampling_num * note[2])
                if note[0]==0:
                    frame = frame * 2 # beat strength는 절대값?
                # new_note = [[note[0], note[1], note[4], 1]] + [[note[0], note[1], note[4], 0]]*(frame-1) # [part_idx, pitch, beat strengh, is_onset]
                new_note = [[note[0], note[1], 1]] + [[note[0], note[1], 0]]*(frame-1)
                new_measure += new_note
                
            dynamic_template = self.dynamic_templates[new_measure[0][0]] #part_idx
            dynamic_template = dynamic_template * (len(new_measure)//len(dynamic_template))
            new_measure_with_dynamic = []
            # print(len(new_measure), len(dynamic_template))
            for i, frame in enumerate(new_measure):
                if len(new_measure) == len(dynamic_template):
                    new_frame = frame+[dynamic_template[i]]
                    new_measure_with_dynamic.append(new_frame)
                else:
                    ValueError('len(new_measure) != len(dynamic_template)')
            new_data.append(new_measure_with_dynamic)

        return new_data
    
    def modify_pitch(self, target_list):
        ratio = 0.3
        modified_list = []
        for note in target_list:
            if random.random() < ratio:
                pitch = random.choice(self.vocab['pitch'][3:])
            else:
                pitch = note[1]
            new_note = [note[0], pitch, note[2], note[3]]
            modified_list.append(new_note)
 
        return modified_list    

    def make_dynamic_template(self):
        whole_dynamic_template = []
        for part_idx in range(len(self.parts)):
            frame = self.beat_sampling_num
            if part_idx == 0:
                # 세종실록악보 : 2.5/1.5, 1.5/1/1.5
                dynamic_mapping = {0.0: 'strong', 2.5: 'strong', 1.5: 'middle', \
                    0.5: 'weak', 1.0: 'weak', 2.0: 'weak', 3.0: 'weak',3.5: 'weak'}
                frame = frame * 2 #세종실록인 경우는 frame이 두 배!
            
            elif part_idx == 1:
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
            whole_measure_len = int(frame * self.offset_list[part_idx])
            dynamic_list = []
            for i in range(whole_measure_len):
                dynamic = dynamic_mapping.get(float(i) / frame, 'pad')
                dynamic_list.append(dynamic)
            whole_dynamic_template.append(dynamic_list)
        return whole_dynamic_template
        
        
    def _get_source_and_target(self, front_part_idx, back_part_idx, measure_number):

        original_source_list = []
        for item in self.converted_features[measure_number][front_part_idx]:
            if isinstance(item, list):
                original_source_list.extend(item)
            else:
                original_source_list.append(item)
        original_target_list = []
        for item in self.converted_features[measure_number][back_part_idx]:
            if isinstance(item, list):
                original_target_list.extend(item)
            else:
                original_target_list.append(item)

        if self.use_pitch_modification and not self.is_valid:
            target_original_target_list = self.modify_pitch(original_target_list)
    
        source = [self.tokenizer(note_feature) for note_feature in original_source_list]
        target = [self.tokenizer(note_feature) for note_feature in original_target_list]

        return source, target
    
    def get_processed_feature(self, front_part_idx, back_part_idx, measure_number):
        source, target = self._get_source_and_target(front_part_idx, back_part_idx, measure_number)
        decoder_input = [[note[0], note[3]] for note in target]
        target = [[note[1], note[2]] for note in target]
        
        return torch.LongTensor(source), torch.LongTensor(decoder_input), torch.LongTensor(target)
    
    def __getitem__(self, idx):
        if self.is_valid:
            front_part_idx, back_part_idx, measure_number = self.valid_index[idx]
        else:    
            front_part_idx = random.choice(range(len(self.parts)-1))
            back_part_idx = random.choice(range(len(self.parts)-1))
            measure_number = random.randint(0, len(self.sliced_features[0])-1 - 2) #하드코딩이긴 함...
        src, tgt, dec_input = self.get_processed_feature(front_part_idx, back_part_idx, measure_number)
        return src, tgt, dec_input
    

class SamplingScore2(SamplingScore):
    def __init__(self, xml_path='/home/danbi/userdata/DANBI/gugakwon/Yeominrak/0_edited.musicxml', \
        slice_measure_num = 2, sample_len = 1000, is_valid=False, use_pitch_modification=False, beat_sampling_num=6) -> None:
        super().__init__(xml_path, slice_measure_num, sample_len, is_valid, use_pitch_modification)

    def convert_to_sampling(self, data):
        new_data = []
        for measure in data:
            new_measure = []
            for note in measure:
                frame = int(self.beat_sampling_num * note[2])
                if note[0]==0:
                    frame = frame * 2 # beat strength는 절대값?
                # new_note = [[note[0], note[1], note[4], 1]] + [[note[0], note[1], note[4], 0]]*(frame-1) # [part_idx, pitch, beat strengh, is_onset]
                new_note = [[note[0], note[1], 1]] + [[note[0], 'sustain', 0]]*(frame-1)
                new_measure += new_note
                
            dynamic_template = self.dynamic_templates[new_measure[0][0]] #part_idx
            dynamic_template = dynamic_template * (len(new_measure)//len(dynamic_template))
            new_measure_with_dynamic = []
            # print(len(new_measure), len(dynamic_template))
            for i, frame in enumerate(new_measure):
                if len(new_measure) == len(dynamic_template):
                    new_frame = frame+[dynamic_template[i]]
                    new_measure_with_dynamic.append(new_frame)
                else:
                    ValueError('len(new_measure) != len(dynamic_template)')
            new_data.append(new_measure_with_dynamic)

        return new_data
    


    def get_processed_feature(self, front_part_idx, back_part_idx, measure_number):
        source, target = self._get_source_and_target(front_part_idx, back_part_idx, measure_number)
        # src = [part_idx, pitch, dynamics]
        decoder_input = [[note[0], note[3]] for note in target] # [part_idx, dynamics]
        target = [note[1] for note in target] # pitch만!
        
        return torch.LongTensor(source), torch.LongTensor(decoder_input), torch.LongTensor(target)



# ------------------ CNN-based ------------------ #

class CNNScore(SamplingScore):
    def __init__(self, xml_path='0_edited.musicxml', \
        slice_measure_num = 2, sample_len = 1000, is_valid=False, use_pitch_modification=False, beat_sampling_num=6) -> None:
        super().__init__(xml_path, slice_measure_num, sample_len, is_valid, use_pitch_modification)
        self.tokenizer = ModifiedTokenizer(self.parts)
    
    def get_processed_feature(self, front_part_idx, back_part_idx, measure_number):
        source, target = self._get_source_and_target(front_part_idx, back_part_idx, measure_number)
        # print(f"target={target}")
        #source/target's shape = [part_idx, pitch, is_onset, dynamic]
        fixed_src = [[note[0], note[1]] for note in source]
        fixed_tgt = [[note[0], note[1]] for note in target]
        
        return torch.LongTensor(fixed_src), torch.LongTensor(fixed_tgt)
    
    def __getitem__(self, idx):
        if self.is_valid:
            front_part_idx, back_part_idx, measure_number = self.valid_index[idx]
        else:    
            front_part_idx = random.choice(range(len(self.parts)-1))
            back_part_idx = random.choice(range(len(self.parts)-1))
            measure_number = random.randint(0, len(self.sliced_features[0])-1 - 2) #하드코딩이긴 함...
        src, tgt = self.get_processed_feature(front_part_idx, back_part_idx, measure_number)
        return src, tgt
    
class BasicWork:
    def __init__(self, xml_path='0_edited.musicxml') -> None:
        self.xml_path = xml_path
        self.score = converter.parse(xml_path)
        self.parts = [Part(part, i) for i, part in enumerate(self.score.parts)]
        self.tokenizer = Tokenizer(self.parts)
        self.vocab = self.tokenizer.vocab
        self.vocab_size_dict = self.tokenizer.vocab_size_dict
        self.offset_list = [part_list.measure_duration for part_list in self.parts]
        self.measure_offset_vocab = []
        self.measure_features = [self.get_feature(i) for i in range(len(self.parts))]
        max_len = max([len(part) for part in self.measure_features])
        self.measure_features = [part + [[] for _ in range(max_len - len(part))] for part in self.measure_features]
        self.part_templates = self._get_part_templates()
        
    def get_feature(self, part_idx):
      part = self.parts[part_idx]
      measure_set = []
      for measure_idx, measure in enumerate(part.measures):
          each_measure = []
          for note in measure:
              each_measure.append([part_idx, note.pitch, note.duration, note.measure_offset, note.dynamic])
          if part_idx in [2,3,4,5,6,7] and sum([note[2] for note in each_measure]) < 10.0:
              each_measure = []
          measure_set.append(each_measure) 
      return measure_set      

    def _get_part_templates(self):
        whole_part_templates = []
        for part in self.measure_features:
            part_templates = []
            for measure in part:
                temp = [note[2] for note in measure]
                part_templates.append(temp)
            # part_templates = set(tuple(temp) for temp in part_templates)
            whole_part_templates.append(part_templates)
        return whole_part_templates

if __name__ == "__main__": 
    train_dataset = OrchestraScoreSeq(is_valid = False)
    val_dataset = ShiftedAlignedScore(is_valid = True, slice_measure_num=10)

    for i in range(len(val_dataset)):
        out = val_dataset[i]
        if len(out[0]) < 4 or len(out[1]) < 4 or len(out[2]) < 4:
          print(f"{i}: {len(out[0])}, {len(out[1])}, {len(out[2])}")

    for i in range(len(train_dataset)):
        out = train_dataset[i]
        if len(out[0]) < 4 or len(out[1]) < 4 or len(out[2]) < 4:
          print(f"{i}: {len(out[0])}, {len(out[1])}, {len(out[2])}")

