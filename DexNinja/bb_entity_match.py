import re


def bbox_matching(bbox_entity, bbox_line):
    x1_1, x1_2 = bbox_entity[0]['x'], bbox_entity[2]['x']
    y1_1, y1_2 = bbox_entity[1]['y'], bbox_entity[2]['y']
    x2_1, x2_2 = bbox_line[0]['x'], bbox_line[2]['x']
    y2_1, y2_2 = bbox_line[1]['y'], bbox_line[2]['y']
    if (x1_1<=x2_1 and x1_2>=x2_1) or (x1_1<=x2_2 and x1_2>=x2_2):
        return True
    if (y1_1<=y2_1 and y1_2>=y2_1) or (y1_1<=y2_2 and y1_2>=y2_2):
        return True
    if (y1_1>=y2_1 and y1_2<=y2_2) or (x1_1>=x2_1 and x1_2<=x2_2):
        return True
    return False

def bb_match(entity, bbox, page, line_no):
    # y1, y2 = bbox[0]['x'], bbox[2]['x']
    for idx in range(line_no+1, len(page['lines'])):
        sent = page['lines'][idx]['content']
        # bbox_y1, bbox_y2 = page['lines'][idx]['polygon'][0]['x'], page['lines'][idx]['polygon'][2]['x']
        # if (y1<bbox_y1 and y2>bbox_y1) or (y1<bbox_y1 and y2>bbox_y1): ## Need to edit to improve matching
        if bbox_matching(bbox, page['lines'][idx]['polygon']):
            matches = entity_value_match(entity, sent)
            if matches:
                return matches
    return []

        
entity_mappings, rxs = {}, {}
entity_mappings['INVOICE NUMBER'] = ['Invoice No', 'Invoice Number']
entity_mappings['DATE'] = ['DATE', 'DT']
entity_mappings['EMAIL'] = ['MAIL']
entity_mappings['WEBSITE'] = ['WEB', 'URL']
entity_mappings['TOTAL AMOUNT'] = ["TOTALEXCLVAT","SUBTOTAL","GOODSTOTAL","NETAMOUNT","NETTGOODSVALUE","NETTOTAL","NETT","TOTALGOODS",
                    " TOTALGOODSVALUE","AMOUNTEXCLVAT","GOODSANDSERVICES","GOODSSUBTOTAL","GOODSTOTALEXCLVAT","GOODSTOTAL","GOODSVAL","GOODS",
                    "GOODSVALUE","NETAMOUNT£","NETGOODS","NETAMOUNTGBP","NETTGOODS","NETTOTAL","NETTTOTAL","NETVALUE","TAXABLEGOODS",
                    "TAXABLESUBTOTAL","TOTALAMOUNTDUEGBP","TOTALEXCLUDINGVAT","TOTALEXVAT","TOTALNETAMOUNT","TOTALNET","TOTALVALUEOFORDEREXVAT",
                    "TOTVALUEOFORDEREXVAT","GOODSAMOUNT","GOODSNETT","NETSALESTOTAL","NET","NETTVALUE","SUBTOTALS","TAXABLEAMOUNT","SUBTOTALS",
                    "TAXABLEAMOUNT","TOTALEXCLUSIVEOFVAT","TOTALSUBJECTTOVAT","TOTALTAXABLEGOODS","GOODSAMOUNT","GOODSNETT","TAXABLEAMOUNT",
                    "TOTALGOODSVALUEEXCLUDINGVAT","TOTALSUBJECTTOVAT","TOTALGOODSVAIUFEXCLUDINGVAT","TOTALGOODSVALUE","GROSSAMOUNT",
                    "TOTALAMOUNT","NETVALUEEXCLVAT","INVOICESUBTOTAL","TOTALAMOUNTOFTHISINVOICE", "TOTAL"]
entity_mappings['TRANSACTION NUMBER'] = ['TRN', 'TRN NUMBER', 'TRN NUM', 'transaction no', 'tranxn no', 'transaction number', 'TXN']
entity_mappings['PURCHASE ORDER'] = ['Purchase Order']

rxs['INVOICE NUMBER'] = ['[#:; ]*([a-z]*[A-Z]*[0-9]+)']
rxs['DATE'] = ['[#:; ]*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})', '([0-9]{1,2}[-,:/ ][A-Z]*[a-z]+[-,:/ ][0-9]{2,4})']
rxs['EMAIL'] = ['[#:; ]*([0-9a-zA-Z]*@[a-z]*.[a-z]*)']
rxs['WEBSITE'] = ['[#:; ]*(www\.[a-z0-9A-Z-.]+[\.com|\.in])']
rxs['TOTAL AMOUNT'] = ['[#:; ]*([$#]*[0-9]+[0-9,.]*)']
rxs['TRANSACTION NUMBER'] = ['[#:; ]*([a-z]*[A-Z]*[0-9]+)']
rxs['PURCHASE ORDER'] = ['[#:; ]*([a-z]*[A-Z]*[0-9]+)']


def match(pat, line):
    pat = pat.lower()
    line = line.lower()
    if pat in line:
        return True
    return False

def entity_match(line):
    entities = []
    for entity in entity_mappings.keys():
        for ent_pat in entity_mappings[entity]:
            if match(ent_pat, line):
                entities.append(entity)
    return entities

def entity_value_match(entity, splitted_line):
    patterns = rxs[entity]
    values = []
    for pat in patterns:
        values.extend(re.findall(pat, splitted_line))
    return values
