from fractions import Fraction
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
BEAT_POSITION = ['|', '\n']+ ['beat:0', 'beat:1/2', 'beat:1/3', 'beat:1/4',  'beat:1/6',
                              'beat:1/9', 'beat:2/3', 'beat:2/9', 'beat:3/4', 'beat:5/6',
                              'beat:7/9', 'beat:8/9']

PITCH = [ '하하배임','하배황', '하배태', '하배중', '하배임', '하배이', '하배남', '하배무',
          '배황', '배태', '배협', '배고', '배중', '배임', '배남', '배무', '배응',
          '황', '태', '협', '고', '중', '임', '이', '남', '무', 
          '청황', '청태', '청협', '청고', '청중', '청임', '청남', '청무',
          '중청황']
PART = ['daegeum', 'piri', 'haegeum', 'gayageum', 'geomungo', 'ajaeng']

DURATION = [ 0,1,2,3,4,5,6,7,8,9,10,12,21,23, Fraction(1, 6), Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(7, 12), Fraction(2, 3), Fraction(3, 4), Fraction(5, 6),
            Fraction(7, 6), Fraction(4, 3), Fraction(3, 2), Fraction(5, 3), Fraction(7, 4), Fraction(11, 6), Fraction(7, 3), Fraction(5, 2),
            Fraction(8, 3), Fraction(17, 6), Fraction(10, 3), Fraction(11, 3), Fraction(23, 6), Fraction(13, 3), Fraction(9, 2), Fraction(14, 3), 
            Fraction(17, 3), Fraction(35, 6), Fraction(19, 3), Fraction(20, 3), Fraction(23, 3), Fraction(59, 6), Fraction(38, 3),
            Fraction(7, 2), Fraction(11, 4), Fraction(17, 2), Fraction(13, 1), Fraction(15, 4), Fraction(19, 4),Fraction(11, 2), Fraction(5, 4), 
            Fraction(15, 2), Fraction(13, 6), Fraction(19, 6), Fraction(29, 3), Fraction(32, 3), Fraction(44, 3), Fraction(23, 2), Fraction(9, 4), 
            Fraction(31, 6), Fraction(29, 6), Fraction(34, 3), Fraction(22, 3), Fraction(16, 3), Fraction(53, 6), Fraction(77, 6)]

BEAT = ['0', '1/2', '1/3', '1/4', '1/6', '2/3', '3/4', '5/6']

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

