from xml.etree.ElementTree import XMLParser, ElementTree, Element, indent
from pathlib import Path
import os
import sys
import random

ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890'

def truthy(s):
    if s.lower() not in ['false', 'f', '']:
        return True
    else:
        return False

'''
This method reads an XML tree, finds elements with tag {tag}, then replaces them with {__n} occurrences
of elements with tag {__tag} and attributes defined by the original element's attributes

e.g. with {tag=stl_insert}

<stl_insert __n='3' __tag='mesh' name='box{i:03d}' mesh='box{i:03d}' euler='1.57 0 0' />

becomes

<mesh name='box000' mesh='box000' euler='1.57 0 0' />
<mesh name='box001' mesh='box001' euler='1.57 0 0' />
<mesh name='box002' mesh='box002' euler='1.57 0 0' />
'''
def parse_xml(f_xml, template_tag, new_f_xml, tag_replacers):
    if not f_xml.exists():
        raise Exception("XML file does not exist")

    tree = ElementTree()
    tree.parse(str(f_xml))

    tag_els = tree.findall(f'.//{template_tag}/..')

    gen_tags = {tag: dict() for tag in tag_replacers.keys()}

    for el in tag_els:
        els = el.findall(f'./{template_tag}')

        for e in els:
            attrib = e.attrib.copy()
            __tag = attrib.pop('__tag')
            __n = int(attrib.pop('__n', '64'))
            __id = attrib.pop('__id', ''.join([random.choice(ALPHABET) for _ in range(16)]))

            if __tag is None:
                raise Exception('No __tag attribute found in element')
            
            for tag_replacer in tag_replacers.values():
                tag_replacer.next(attrib)
            
            el.remove(e)

            for i in range(__n):
                a = attrib.copy()

                for key,val in a.items():
                    a[key] = str.format(val, i=i)
                
                for tag_replacer in tag_replacers.values():
                    tag_replacer.modify_attr(a, attrib)
                
                new_e = Element(__tag, a)
                el.append(new_e)
            
            for tag,replacer in tag_replacers.items():
                if replacer.value is not None:
                    gen_tags[tag][__id] = [replacer.descriptor, replacer.value]

    indent(tree, '    ')
    tree.write(str(new_f_xml))

    return gen_tags
