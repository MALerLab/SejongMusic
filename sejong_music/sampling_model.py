import torch
import music21
from collections import defaultdict, Counter

from .yeominrak_processing import AlignedScore, Tokenizer, OrchestraScore
from .utils import convert_note_to_sampling, convert_onset_to_sustain_token, make_dynamic_template, convert_dynamics_to_integer
from .yeominrak_processing import TestScore

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
class SamplingScore(OrchestraScore):
  def __init__(self, xml_path='0_edited.musicxml', 
               valid_measure_num=[i for i in range(93, 104)], 
               slice_measure_num=2, is_valid=False, 
               use_pitch_modification=False, 
               pitch_modification_ratio=0, 
               min_meas=3, 
               max_meas=6, 
               feature_types=[],
               sampling_rate=2,
               is_sep=False) -> None:
    super().__init__(xml_path, valid_measure_num, slice_measure_num, is_valid, use_pitch_modification, pitch_modification_ratio, min_meas, max_meas, feature_types, is_sep=False)
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

  def _get_feature_by_note(self, note, part_idx):
    features = [part_idx+1] # orchestra score starts from 1
    # if 'pitch' in self.tokenizer.key_types:
    features.append(note.pitch) 
    # if 'duration' in self.tokenizer.key_types:
    features.append(note.duration)
    if 'offset' in self.tokenizer.key_types:
        features.append(note.measure_offset)
    if self.is_sep:
        offsets = make_offset_set(note.measure_offset)  # [daegang, jeonggan, beat]
        features += offsets
    if 'dynamic' in self.tokenizer.key_types:
        features.append(self.dynamic_template.get(note.measure_offset, 'none'))
    if 'measure_idx' in self.tokenizer.key_types:
        features.append(note.measure_number)
    if 'measure_change' in self.tokenizer.key_types:
        changed = 1 if note.measure_offset == 0.0 else 0
        features.append(changed)
    return features




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

class QkvRollSeq2seq(QkvAttnSeq2seq):
  def __init__(self, vocab, config):
    super().__init__(vocab, config)    
    self.encoder.return_reduced_bidirectional = False
    if hasattr(config, 'sampling_rate'):
      self.sampling_rate = config.sampling_rate 

    self.converter = RollConverter(self.tokenizer, self.sampling_rate)
    self.decoder = QkvRollDecoder(self.vocab_size_dict, self.config)
    self.dynamic_template_list = make_dynamic_template(beat_sampling_num=self.sampling_rate)
    self.dynamic_template_list = [convert_dynamics_to_integer(d) for d in self.dynamic_template_list]

  def forward(self, src, dec_input):
    enc_out, enc_last_hidden = self.encoder(src)
    dec_out = self.decoder(src, dec_input, enc_out, enc_last_hidden)
    return dec_out
  
  def make_dec_input(self, part_idx, measure_len):
    dynamic_template = self.dynamic_template_list[part_idx]
    num_frames = len(dynamic_template) * measure_len

    output = np.zeros((num_frames, 4), dtype=np.int32)
    output[:, 0] = part_idx
    output[:, 1] = 0 # add dummy note
    output[:, 2] = dynamic_template * measure_len
    output[:, 3] = np.arange(num_frames)

    output = torch.LongTensor([self.tokenizer(frame)  for frame in output] )
    return output[:, [0, 2, 3]]
    

  @torch.inference_mode()
  def inference(self, src, target_idx):
    dev = src.device
    encode_out, enc_last_hidden = self.encoder(src.unsqueeze(0)) 
    src_part_idx = int(src[0][0])

    measure_len = int(src.shape[0] / self.sampling_rate / MEAS_LEN_BY_IDX[src_part_idx])
    dec_input = self.make_dec_input(target_idx, measure_len).to(dev)
    dec_input = self.decoder._get_embedding(dec_input)
    decode_out, last_hidden = self.decoder.rnn(dec_input.unsqueeze(0), enc_last_hidden)
    attention_vectors, attention_weight = self._get_attention_vector(encode_out, decode_out)
    combined_value = torch.cat([decode_out, attention_vectors], dim=-1)
    combined_value = combined_value.squeeze(0)
    logit = self.decoder.proj(combined_value)
    selected_pitch = torch.argmax(logit, dim=-1)

    selected_token = [[target_idx, int(p)] for p in selected_pitch]

      
    return self.converter([note[:2] for note in src]), self.converter(selected_token), (attention_weight, selected_token, selected_token[-len(self.dynamic_template_list[target_idx])*2:-len(self.dynamic_template_list[target_idx])])


class QkvRollDecoder(QkvAttnDecoder):
  def __init__(self, vocab_size_dict: dict, param):
    self.param = param
    super().__init__(vocab_size_dict, param)
    self.rnn = nn.GRU(self.emb.total_size, self.hidden_size, num_layers=param.gru.num_layers, batch_first=True, bidirectional=True, dropout=param.gru.dropout)
    self.query = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.key = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.value = nn.Linear(self.hidden_size * 2, self.hidden_size)
    self.dropout = nn.Dropout(param.gru.dropout)

  def _make_embedding_layer(self): # 
    vocab_size_dict = copy.copy(self.vocab_size_dict)
    vocab_size_dict.pop('pitch')
    self.emb = MultiEmbedding(vocab_size_dict, self.param.emb)


  def _apply_softmax(self, logit):
    pitch_output = torch.softmax(logit, dim=-1)
    return pitch_output

  def _make_projection_layer(self):
    # total_vocab_size = sum([x for x in self.vocab_size_dict.values()])
    self.proj = nn.Linear(self.hidden_size * 3, self.vocab_size[1]) # pitch + is_onset
    
  def forward(self, src, dec_input, enc_hidden_state_by_t, enc_out):
    if isinstance(dec_input, PackedSequence):
      emb = self._get_embedding(dec_input.data)
      emb = PackedSequence(emb, dec_input[1], dec_input[2], dec_input[3])
      dec_hidden_state_by_t, dec_out = self.rnn(emb, enc_out)
      mask = pad_packed_sequence(src, batch_first=True)[0][... , 1:2] != 0
      # out = pad_packed_sequence(out, batch_first=True)[0]
      encoder_hidden_states, source_lens = pad_packed_sequence(enc_hidden_state_by_t, batch_first=True)
      decoder_hidden_states, target_lens = pad_packed_sequence(dec_hidden_state_by_t, batch_first=True)
      query = self.query(decoder_hidden_states)
      key = self.key(encoder_hidden_states)
      value = self.value(encoder_hidden_states)
      
      attention_score = torch.bmm(key, query.transpose(1, 2))
      attention_score = attention_score.masked_fill(mask == 0, -float("inf"))
      attention_weight = torch.softmax(attention_score, dim=1)
      attention_vec = torch.bmm(attention_weight.transpose(1, 2), value)
      dec_output = torch.cat([decoder_hidden_states, attention_vec], dim=-1)
      dec_output = self.dropout(dec_output)
      
      logit = self.proj(dec_output)
      prob = self._apply_softmax(logit)
      prob = pack_padded_sequence(prob, target_lens, batch_first=True, enforce_sorted=False)
      
    else:
      raise NotImplementedError
    return prob, attention_weight



  
  #-------------Transformer -----------------#


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

