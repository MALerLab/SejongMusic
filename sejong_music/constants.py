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

POSITION = ['|', '\n']+ [f":{i}" for i in range(0, 16)]
PITCH = [ '하하배임','하배황', '하배태', '하배중', '하배임', '하배이', '하배남', '하배무',
          '배황', '배태', '배협', '배고', '배중', '배임', '배남', '배무', '배응',
          '황', '태', '협', '고', '중', '임', '이', '남', '무', 
          '청황', '청태', '청협', '청고', '청중', '청임', '청남', '청무',
          '중청황']
PART = ['daegeum', 'piri', 'haegeum', 'gayageum', 'geomungo', 'ajaeng']

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

