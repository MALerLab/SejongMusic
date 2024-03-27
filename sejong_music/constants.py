MEAS_LEN_BY_IDX = [8.0, 8.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]
DYNAMIC_MAPPING = [
  {0.0: 'strong', 2.5: 'strong', 1.5: 'middle'},
  # {0.0: 'strong', 3.0: 'strong', 6.0: 'middle'},
  {0.0: 'strong', 5.0: 'strong', 3.0: 'middle'},
  {0.0: 'strong', 6.0: 'strong', 3.0: 'middle', 8.0: 'middle'},
  {0.0: 'strong', 6.0: 'strong', 3.0: 'middle', 8.0: 'middle'},
  {0.0: 'strong', 6.0: 'strong', 3.0: 'middle', 8.0: 'middle'},
  {0.0: 'strong', 6.0: 'strong', 3.0: 'middle', 8.0: 'middle'},
  {0.0: 'strong', 5.0: 'strong', 3.0: 'middle', 7.0: 'middle'},
  {0.0: 'strong', 5.0: 'strong', 3.0: 'middle', 7.0: 'middle'}
]

def get_dynamic(current_beat, part_idx):
    dynamic_mapping = DYNAMIC_MAPPING[part_idx]
    dynamic = dynamic_mapping.get(current_beat, 'weak')
    return dynamic
  

def get_dynamic_template_for_orch():
  dynamic_template = {x/2: 'weak' for x in range(0, 90, 3)}
  dynamic_template[0.0] =  'strong'
  dynamic_template[15.0] =  'strong'
  dynamic_template[9.0] =  'middle'
  dynamic_template[21.0] =  'middle'

  return dynamic_template