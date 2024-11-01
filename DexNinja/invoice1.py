import json
from bb_entity_match import entity_match, entity_value_match
from bb_entity_match import bb_match

with open("data/json/ABM_ducument_2_page_6.json", "rb") as f:
    full_text = json.load(f)


# entity_list = ['INVOICE NUMBER', 'DATE', 'EMAIL', 'TOTAL AMOUNT', 'TRANSACTION NUMBER', 'PURCHASE ORDER']
entity_list = ['DATE', 'EMAIL', 'WEBSITE']
result = {}
for page in full_text['pages']:
    for line_no, line_dict in enumerate(page['lines']):
        line = line_dict['content']
        entities = entity_match(line)
        for ent in entities:
            splitted_line = line.replace(ent, "")
            matched_values = entity_value_match(ent, splitted_line)
            if ent in result.keys():
                result[ent].extend(matched_values)
            else:
                result[ent] = matched_values

            if not result[ent]:
                bbox = line_dict['polygon']
                result[ent] = bb_match(ent, bbox, page, line_no)

for ent in entity_list:
    if ent not in result.keys()or not result[ent]:
        for page in full_text['pages']:
            for line_no, line_dict in enumerate(page['lines']):
                if ent in result.keys():
                    result[ent].extend(entity_value_match(ent, line_dict['content']))
                else:
                    result[ent] = entity_value_match(ent, line_dict['content'])
print(result)